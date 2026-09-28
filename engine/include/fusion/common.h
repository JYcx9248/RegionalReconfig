// Common types and utilities shared by the builder, the engine and the tools.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion {

constexpr uint32_t kInvalidId = 0xFFFFFFFFu;
// SSD read granularity. Raw vectors are packed into pages of this size.
constexpr uint32_t kPageSize = 4096;

// Element type of the raw vectors (the three billion-scale benchmarks use
// uint8 = SIFT1B/BigANN, int8 = SPACEV1B, float = DEEP1B).
enum class DType : uint8_t { kUInt8 = 0, kInt8 = 1, kFloat = 2 };

size_t DTypeSize(DType t);
const char* DTypeName(DType t);
DType ParseDType(const std::string& s);

// Calls f(T{}) with T = uint8_t / int8_t / float according to `t`.
template <class F>
decltype(auto) DispatchDType(DType t, F&& f) {
  switch (t) {
    case DType::kUInt8: return f(uint8_t{});
    case DType::kInt8: return f(int8_t{});
    case DType::kFloat: return f(float{});
  }
  throw std::runtime_error("unknown dtype");
}

template <class T> constexpr DType DTypeOf();
template <> constexpr DType DTypeOf<uint8_t>() { return DType::kUInt8; }
template <> constexpr DType DTypeOf<int8_t>() { return DType::kInt8; }
template <> constexpr DType DTypeOf<float>() { return DType::kFloat; }

std::string StrFormat(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

[[noreturn]] void ThrowError(const char* file, int line, const std::string& msg);

#define FUSION_CHECK(cond, ...)                                              \
  do {                                                                       \
    if (!(cond)) ::fusion::ThrowError(__FILE__, __LINE__,                    \
                                      ::fusion::StrFormat(__VA_ARGS__));     \
  } while (0)

// Timestamped log line on stderr.
void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}
  void Reset() { start_ = std::chrono::steady_clock::now(); }
  double Us() const {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start_)
        .count();
  }
  double Ms() const { return Us() / 1000.0; }
  double Sec() const { return Us() / 1e6; }

 private:
  std::chrono::steady_clock::time_point start_;
};

// Aligned heap memory (for O_DIRECT buffers and SIMD-friendly arrays).
struct FreeDeleter {
  void operator()(void* p) const { std::free(p); }
};
using AlignedBuffer = std::unique_ptr<uint8_t[], FreeDeleter>;
AlignedBuffer AllocAligned(size_t alignment, size_t bytes);

inline uint64_t CeilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
inline uint64_t RoundUp(uint64_t a, uint64_t b) { return CeilDiv(a, b) * b; }
inline uint32_t NextPow2(uint32_t v) {
  uint32_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

// "1,2,4" -> {1,2,4}
std::vector<uint64_t> ParseUintList(const std::string& s);
std::string JoinPath(const std::string& a, const std::string& b);
bool FileExists(const std::string& path);
void MakeDirs(const std::string& path);
uint64_t FileSize(const std::string& path);

// Binary file helpers (whole-array read/write with error checking).
void WriteFile(const std::string& path, const void* data, size_t bytes);
void AppendToFile(FILE* f, const void* data, size_t bytes, const std::string& path_for_errors);
std::vector<uint8_t> ReadFile(const std::string& path);

template <class T>
void WriteVector(FILE* f, const std::vector<T>& v, const std::string& path) {
  AppendToFile(f, v.data(), v.size() * sizeof(T), path);
}

}  // namespace fusion
