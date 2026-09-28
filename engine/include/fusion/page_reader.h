// Batched 4 KB page reads from the raw-vector file (the SSD tier).
//
// Backends: io_uring (when built with liburing), Linux native AIO (raw syscalls, no library
// needed; what DiskANN and SPANN use), and synchronous pread as a portable fallback. The
// file is opened with O_DIRECT by default so reads really hit the device (paper Section 3.4).
#pragma once

#include <memory>
#include <string>

#include "fusion/common.h"

namespace fusion {

enum class IoBackend { kAuto, kUring, kAio, kPread };
IoBackend ParseIoBackend(const std::string& s);
const char* IoBackendName(IoBackend b);

class PageFile {
 public:
  // Falls back to buffered I/O (with a warning) if O_DIRECT is not supported.
  PageFile(const std::string& path, uint32_t page_size, bool direct);
  ~PageFile();
  PageFile(const PageFile&) = delete;
  PageFile& operator=(const PageFile&) = delete;

  int fd() const { return fd_; }
  uint32_t page_size() const { return page_size_; }
  uint64_t num_pages() const { return num_pages_; }
  bool direct() const { return direct_; }

 private:
  int fd_ = -1;
  uint32_t page_size_;
  uint64_t num_pages_ = 0;
  bool direct_ = false;
};

// One reader per worker thread (not thread-safe).
class PageReader {
 public:
  virtual ~PageReader() = default;
  // Reads pages[i] into bufs[i] (page_size bytes each, 4 KB-aligned) for i < count and
  // returns when all reads have completed. Throws on I/O errors.
  virtual void Read(const uint32_t* pages, uint8_t* const* bufs, uint32_t count) = 0;
  virtual IoBackend backend() const = 0;
};

// kAuto tries io_uring, then AIO, then pread.
std::unique_ptr<PageReader> MakePageReader(const PageFile& file, IoBackend backend,
                                           uint32_t queue_depth);

}  // namespace fusion
