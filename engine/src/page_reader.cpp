#include "fusion/page_reader.h"

#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <vector>

#ifdef FUSION_HAVE_LIBURING
#include <liburing.h>
#endif

namespace fusion {

IoBackend ParseIoBackend(const std::string& s) {
  if (s == "auto") return IoBackend::kAuto;
  if (s == "uring" || s == "io_uring") return IoBackend::kUring;
  if (s == "aio" || s == "libaio") return IoBackend::kAio;
  if (s == "pread" || s == "sync") return IoBackend::kPread;
  throw std::runtime_error("unknown I/O backend '" + s + "' (auto, uring, aio, pread)");
}

const char* IoBackendName(IoBackend b) {
  switch (b) {
    case IoBackend::kAuto: return "auto";
    case IoBackend::kUring: return "io_uring";
    case IoBackend::kAio: return "aio";
    case IoBackend::kPread: return "pread";
  }
  return "?";
}

PageFile::PageFile(const std::string& path, uint32_t page_size, bool direct)
    : page_size_(page_size) {
  int flags = O_RDONLY;
  if (direct) flags |= O_DIRECT;
  fd_ = ::open(path.c_str(), flags);
  direct_ = direct;
  if (fd_ < 0 && direct && errno == EINVAL) {
    Log("warning: O_DIRECT not supported for %s; falling back to buffered I/O "
        "(results will include page-cache effects)",
        path.c_str());
    fd_ = ::open(path.c_str(), O_RDONLY);
    direct_ = false;
  }
  FUSION_CHECK(fd_ >= 0, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  num_pages_ = FileSize(path) / page_size;
}

PageFile::~PageFile() {
  if (fd_ >= 0) ::close(fd_);
}

namespace {

void CheckRead(long long res, uint32_t page_size, uint32_t page) {
  if (res == static_cast<long long>(page_size)) return;
  if (res < 0) {
    throw std::runtime_error(StrFormat("read of page %u failed: %s", page,
                                       std::strerror(static_cast<int>(-res))));
  }
  throw std::runtime_error(StrFormat("short read of page %u (%lld bytes)", page, res));
}

class PreadReader : public PageReader {
 public:
  explicit PreadReader(const PageFile& f) : fd_(f.fd()), ps_(f.page_size()) {}
  void Read(const uint32_t* pages, uint8_t* const* bufs, uint32_t count) override {
    for (uint32_t i = 0; i < count; ++i) {
      size_t got = 0;
      const off_t off = static_cast<off_t>(pages[i]) * ps_;
      while (got < ps_) {
        ssize_t r = ::pread(fd_, bufs[i] + got, ps_ - got, off + static_cast<off_t>(got));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) CheckRead(r < 0 ? -errno : static_cast<long long>(got), ps_, pages[i]);
        got += static_cast<size_t>(r);
      }
    }
  }
  IoBackend backend() const override { return IoBackend::kPread; }

 private:
  int fd_;
  uint32_t ps_;
};

// Linux native AIO through raw syscalls (no libaio dependency).
class AioReader : public PageReader {
 public:
  AioReader(const PageFile& f, uint32_t depth)
      : fd_(f.fd()), ps_(f.page_size()), depth_(depth), iocbs_(depth), ptrs_(depth), events_(depth) {
    if (syscall(__NR_io_setup, depth_, &ctx_) < 0) {
      throw std::runtime_error(StrFormat("io_setup failed: %s", std::strerror(errno)));
    }
  }
  ~AioReader() override { syscall(__NR_io_destroy, ctx_); }

  void Read(const uint32_t* pages, uint8_t* const* bufs, uint32_t count) override {
    for (uint32_t done = 0; done < count;) {
      const uint32_t batch = std::min(depth_, count - done);
      for (uint32_t j = 0; j < batch; ++j) {
        iocb& cb = iocbs_[j];
        std::memset(&cb, 0, sizeof(cb));
        cb.aio_fildes = static_cast<uint32_t>(fd_);
        cb.aio_lio_opcode = IOCB_CMD_PREAD;
        cb.aio_buf = reinterpret_cast<uint64_t>(bufs[done + j]);
        cb.aio_nbytes = ps_;
        cb.aio_offset = static_cast<int64_t>(pages[done + j]) * ps_;
        cb.aio_data = done + j;
        ptrs_[j] = &cb;
      }
      for (uint32_t sub = 0; sub < batch;) {
        long r = syscall(__NR_io_submit, ctx_, static_cast<long>(batch - sub), ptrs_.data() + sub);
        if (r < 0) {
          if (errno == EINTR || errno == EAGAIN) continue;
          throw std::runtime_error(StrFormat("io_submit failed: %s", std::strerror(errno)));
        }
        sub += static_cast<uint32_t>(r);
      }
      for (uint32_t got = 0; got < batch;) {
        long r = syscall(__NR_io_getevents, ctx_, static_cast<long>(batch - got),
                         static_cast<long>(batch - got), events_.data(), nullptr);
        if (r < 0) {
          if (errno == EINTR) continue;
          throw std::runtime_error(StrFormat("io_getevents failed: %s", std::strerror(errno)));
        }
        for (long e = 0; e < r; ++e)
          CheckRead(events_[e].res, ps_, pages[static_cast<uint32_t>(events_[e].data)]);
        got += static_cast<uint32_t>(r);
      }
      done += batch;
    }
  }
  IoBackend backend() const override { return IoBackend::kAio; }

 private:
  int fd_;
  uint32_t ps_;
  uint32_t depth_;
  aio_context_t ctx_ = 0;
  std::vector<iocb> iocbs_;
  std::vector<iocb*> ptrs_;
  std::vector<io_event> events_;
};

#ifdef FUSION_HAVE_LIBURING
class UringReader : public PageReader {
 public:
  UringReader(const PageFile& f, uint32_t depth) : fd_(f.fd()), ps_(f.page_size()), depth_(depth) {
    int r = io_uring_queue_init(depth_, &ring_, 0);
    if (r < 0) {
      throw std::runtime_error(StrFormat("io_uring_queue_init failed: %s", std::strerror(-r)));
    }
  }
  ~UringReader() override { io_uring_queue_exit(&ring_); }

  void Read(const uint32_t* pages, uint8_t* const* bufs, uint32_t count) override {
    for (uint32_t done = 0; done < count;) {
      const uint32_t batch = std::min(depth_, count - done);
      for (uint32_t j = 0; j < batch; ++j) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        FUSION_CHECK(sqe != nullptr, "io_uring submission queue full");
        io_uring_prep_read(sqe, fd_, bufs[done + j], ps_,
                           static_cast<uint64_t>(pages[done + j]) * ps_);
        sqe->user_data = done + j;
      }
      int r;
      do {
        r = io_uring_submit_and_wait(&ring_, batch);
      } while (r == -EINTR);
      if (r < 0) throw std::runtime_error(StrFormat("io_uring_submit failed: %s", std::strerror(-r)));
      for (uint32_t got = 0; got < batch;) {
        io_uring_cqe* cqe = nullptr;
        r = io_uring_wait_cqe(&ring_, &cqe);
        if (r == -EINTR) continue;
        if (r < 0) throw std::runtime_error(StrFormat("io_uring_wait_cqe failed: %s", std::strerror(-r)));
        long long res = cqe->res;
        uint32_t page = pages[static_cast<uint32_t>(cqe->user_data)];
        io_uring_cqe_seen(&ring_, cqe);
        CheckRead(res, ps_, page);
        ++got;
      }
      done += batch;
    }
  }
  IoBackend backend() const override { return IoBackend::kUring; }

 private:
  int fd_;
  uint32_t ps_;
  uint32_t depth_;
  io_uring ring_;
};
#endif

}  // namespace

std::unique_ptr<PageReader> MakePageReader(const PageFile& file, IoBackend backend,
                                           uint32_t queue_depth) {
  queue_depth = std::max<uint32_t>(1, queue_depth);
  if (backend == IoBackend::kUring || backend == IoBackend::kAuto) {
#ifdef FUSION_HAVE_LIBURING
    try {
      return std::make_unique<UringReader>(file, queue_depth);
    } catch (const std::exception& e) {
      if (backend == IoBackend::kUring) throw;
    }
#else
    if (backend == IoBackend::kUring)
      throw std::runtime_error("built without liburing; use --io aio or --io pread");
#endif
  }
  if (backend == IoBackend::kAio || backend == IoBackend::kAuto) {
    try {
      return std::make_unique<AioReader>(file, queue_depth);
    } catch (const std::exception& e) {
      if (backend == IoBackend::kAio) throw;
    }
  }
  return std::make_unique<PreadReader>(file);
}

}  // namespace fusion
