#include "fusion/common.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sstream>

namespace fusion {

size_t DTypeSize(DType t) {
  switch (t) {
    case DType::kUInt8: return 1;
    case DType::kInt8: return 1;
    case DType::kFloat: return 4;
  }
  return 0;
}

const char* DTypeName(DType t) {
  switch (t) {
    case DType::kUInt8: return "uint8";
    case DType::kInt8: return "int8";
    case DType::kFloat: return "float";
  }
  return "?";
}

DType ParseDType(const std::string& s) {
  if (s == "uint8" || s == "u8") return DType::kUInt8;
  if (s == "int8" || s == "i8") return DType::kInt8;
  if (s == "float" || s == "float32" || s == "f32") return DType::kFloat;
  throw std::runtime_error("unknown dtype '" + s + "' (expected uint8, int8 or float)");
}

std::string StrFormat(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  va_list ap2;
  va_copy(ap2, ap);
  int n = std::vsnprintf(nullptr, 0, fmt, ap);
  va_end(ap);
  std::string out;
  if (n > 0) {
    out.resize(static_cast<size_t>(n) + 1);
    std::vsnprintf(&out[0], out.size(), fmt, ap2);
    out.resize(static_cast<size_t>(n));
  }
  va_end(ap2);
  return out;
}

void ThrowError(const char* file, int line, const std::string& msg) {
  const char* base = std::strrchr(file, '/');
  throw std::runtime_error(StrFormat("%s:%d: %s", base ? base + 1 : file, line, msg.c_str()));
}

void Log(const char* fmt, ...) {
  static std::mutex mu;
  static const auto t0 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  va_list ap;
  va_start(ap, fmt);
  std::lock_guard<std::mutex> lock(mu);
  std::fprintf(stderr, "[%9.2fs] ", sec);
  std::vfprintf(stderr, fmt, ap);
  std::fputc('\n', stderr);
  va_end(ap);
}

AlignedBuffer AllocAligned(size_t alignment, size_t bytes) {
  void* p = nullptr;
  size_t sz = RoundUp(bytes == 0 ? alignment : bytes, alignment);
  if (posix_memalign(&p, alignment, sz) != 0 || p == nullptr) {
    throw std::bad_alloc();
  }
  return AlignedBuffer(static_cast<uint8_t*>(p));
}

std::vector<uint64_t> ParseUintList(const std::string& s) {
  std::vector<uint64_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::stoull(item));
  }
  return out;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

bool FileExists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

void MakeDirs(const std::string& path) {
  if (path.empty()) return;
  std::string cur;
  std::stringstream ss(path);
  std::string part;
  if (path[0] == '/') cur = "/";
  while (std::getline(ss, part, '/')) {
    if (part.empty()) continue;
    cur += part + "/";
    if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
      throw std::runtime_error("mkdir failed for " + cur + ": " + std::strerror(errno));
    }
  }
}

uint64_t FileSize(const std::string& path) {
  struct stat st;
  FUSION_CHECK(::stat(path.c_str(), &st) == 0, "cannot stat %s: %s", path.c_str(),
               std::strerror(errno));
  return static_cast<uint64_t>(st.st_size);
}

void AppendToFile(FILE* f, const void* data, size_t bytes, const std::string& path) {
  if (bytes == 0) return;
  size_t w = std::fwrite(data, 1, bytes, f);
  FUSION_CHECK(w == bytes, "short write to %s (%zu of %zu bytes): %s", path.c_str(), w, bytes,
               std::strerror(errno));
}

void WriteFile(const std::string& path, const void* data, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s for writing: %s", path.c_str(), std::strerror(errno));
  AppendToFile(f, data, bytes, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

void DropFromPageCache(int fd, uint64_t off, uint64_t len) {
  if (len == 0) return;  // posix_fadvise takes 0 for "to the end of the file"
  static const uint64_t page = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
  const uint64_t a = off / page * page, z = CeilDiv(off + len, page) * page;
  ::posix_fadvise(fd, static_cast<off_t>(a), static_cast<off_t>(z - a), POSIX_FADV_DONTNEED);
}

void SyncAndDropFromPageCache(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  FUSION_CHECK(fd >= 0, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  const bool synced = ::fdatasync(fd) == 0;
  const int err = errno;
  if (synced) DropFromPageCache(fd, 0, FileSize(path));
  ::close(fd);
  FUSION_CHECK(synced, "cannot sync %s: %s", path.c_str(), std::strerror(err));
}

std::vector<uint8_t> ReadFile(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  std::vector<uint8_t> buf(FileSize(path));
  size_t r = buf.empty() ? 0 : std::fread(buf.data(), 1, buf.size(), f);
  std::fclose(f);
  FUSION_CHECK(r == buf.size(), "short read from %s", path.c_str());
  return buf;
}

}  // namespace fusion
