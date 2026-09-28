// GPU filtering backend (paper Sections 3.1-3.2 and 4).
//
//  * PQ codes of all vectors are copied to device memory once and stay pinned there, so a
//    query only ships vector IDs over PCIe, never vector content.
//  * "Contention-free GPU memory management": after the codes are loaded, one pool is
//    allocated and carved into fixed per-worker blocks (query, distance table, candidate IDs,
//    hash table, distance/ID arrays, sort scratch). No allocation or locking on the query path.
//  * Each worker has its own CUDA stream, so concurrent queries overlap on the device.
//
// Kernels
//  LutKernel       one block per sub-space, one thread per centroid: the 256 x m distance table
//  DedupAdcKernel  one thread per candidate ID: lock-free insert into an open-addressing hash
//                  table (atomicCAS; the first inserter owns the ID), then the PQ distance via
//                  the distance table held in shared memory. Duplicates get +inf. Each result
//                  is written as one 64-bit key: (float bits of the distance << 32) | ID.
//  cub::DeviceRadixSort::SortKeys orders the keys; the first n are copied back and split.
//
// Sorting the composite key breaks distance ties by vector ID, exactly like the CPU backend,
// so results do not depend on which thread won the dedup race. (Distances are >= 0, so the
// IEEE-754 bit pattern of a distance orders the same way as its value.)
//
// The paper describes a spinlock-protected hash table and one thread per PQ dimension with a
// coordinator thread; we use a lock-free CAS table and one thread per candidate, which is the
// standard formulation and avoids intra-warp spinlocks. HashId/AdcDistance are shared with the
// CPU backend (fusion/hd_common.h).
//
// rtier (FilterConfig::slot_addressed): a data node holds only the codes of vectors in its
// resident posting lists, one copy each (fusion/pq_store.h). The device buffer then has one
// slot per code of the node's budget, StoreCodes copies newly installed codes into their
// slots, and each query ships a slot array next to its candidate IDs. Keys still carry the
// vector ID, so the order of results (and ties) is the same as with ID addressing.

#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "fusion/filter.h"
#include "fusion/hd_common.h"

namespace fusion {
namespace {

#define FUSION_CUDA_CHECK(expr)                                                        \
  do {                                                                                 \
    cudaError_t err__ = (expr);                                                        \
    if (err__ != cudaSuccess) {                                                        \
      throw std::runtime_error(StrFormat("CUDA error %s at %s:%d: %s",                 \
                                         cudaGetErrorName(err__), __FILE__, __LINE__, \
                                         cudaGetErrorString(err__)));                  \
    }                                                                                  \
  } while (0)

constexpr uint32_t kBlock = 256;
// Distance tables up to 48 KB (m <= 48) are staged in shared memory.
constexpr uint32_t kMaxSharedLutM = 48;

__global__ void LutKernel(const float* __restrict__ query, const float* __restrict__ codebook,
                          uint32_t dsub, float* __restrict__ lut) {
  const uint32_t m = blockIdx.x;
  const uint32_t j = threadIdx.x;  // centroid
  const float* q = query + static_cast<size_t>(m) * dsub;
  const float* c = codebook + (static_cast<size_t>(m) * kPQKsub + j) * dsub;
  float s = 0.f;
  for (uint32_t t = 0; t < dsub; ++t) {
    const float d = q[t] - c[t];
    s += d * d;
  }
  lut[m * kPQKsub + j] = s;
}

// slots: code slot of each candidate (slot-addressed backends), or null (code at id * m).
template <bool kSharedLut>
__global__ void DedupAdcKernel(const uint32_t* __restrict__ cand,
                               const uint32_t* __restrict__ slots, uint32_t n,
                               const uint8_t* __restrict__ codes, uint32_t m,
                               const float* __restrict__ lut_global, uint32_t* table,
                               uint32_t mask, unsigned long long* __restrict__ out_key,
                               uint32_t* unique_count) {
  extern __shared__ float s_lut[];
  const float* lut = lut_global;
  if (kSharedLut) {
    for (uint32_t i = threadIdx.x; i < m * kPQKsub; i += blockDim.x) s_lut[i] = lut_global[i];
    __syncthreads();
    lut = s_lut;
  }
  uint32_t local_unique = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const uint32_t id = cand[i];
    uint32_t h = HashId(id) & mask;
    bool owner = false;
    while (true) {
      const uint32_t prev = atomicCAS(&table[h], kEmptySlot, id);
      if (prev == kEmptySlot) {
        owner = true;
        break;
      }
      if (prev == id) break;  // another thread owns this ID (replicated vector)
      h = (h + 1) & mask;
    }
    float d = __int_as_float(0x7f800000);  // +inf: duplicates sort last
    if (owner) {
      const size_t addr = slots ? slots[i] : id;
      d = AdcDistance(lut, codes + addr * m, m);
      ++local_unique;
    }
    out_key[i] = (static_cast<unsigned long long>(__float_as_uint(d)) << 32) | id;
  }
  if (local_unique) atomicAdd(unique_count, local_unique);
}

// Slot-addressed backends: copies `count` staged codes (m bytes each, in the order of `slots`)
// to their slots.
__global__ void ScatterCodesKernel(const uint8_t* __restrict__ stage,
                                   const uint32_t* __restrict__ slots, uint32_t count, uint32_t m,
                                   uint8_t* __restrict__ codes) {
  const uint64_t total = static_cast<uint64_t>(count) * m;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const uint64_t k = i / m;
    codes[static_cast<uint64_t>(slots[k]) * m + (i - k * m)] = stage[i];
  }
}

// Codes staged per StoreCodes batch (pinned host buffer + device buffer).
constexpr uint32_t kStageCodes = 1u << 16;

// Carves aligned sub-buffers out of a block (or measures the block size when base is null).
class Carver {
 public:
  explicit Carver(char* base) : base_(base) {}
  template <class T>
  T* Take(size_t count) {
    T* p = reinterpret_cast<T*>(base_ ? base_ + off_ : nullptr);
    off_ += RoundUp(std::max<size_t>(count, 1) * sizeof(T), 256);
    return p;
  }
  size_t used() const { return off_; }

 private:
  char* base_;
  size_t off_ = 0;
};

struct GpuShared {
  int device = 0;
  int num_sms = 1;
  uint32_t dim = 0, m = 0, dsub = 0;
  bool slot_mode = false;  // codes addressed by slot (FilterConfig::slot_addressed)
  const uint8_t* d_codes = nullptr;
  const float* d_codebook = nullptr;
};

class GpuFilterWorker : public FilterWorker {
 public:
  // Layout of one worker's device block; also used to size the pool.
  static size_t BlockBytes(const GpuShared& g, uint32_t max_cand, size_t sort_bytes) {
    Carver c(nullptr);
    Layout(&c, g, max_cand, sort_bytes, nullptr);
    return c.used();
  }

  GpuFilterWorker(const GpuShared& g, char* block, uint32_t max_cand, uint32_t max_topn,
                  size_t sort_bytes)
      : g_(g), max_cand_(max_cand), max_topn_(max_topn), sort_bytes_(sort_bytes) {
    Carver c(block);
    Layout(&c, g, max_cand, sort_bytes, this);
    FUSION_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_query_), g.dim * sizeof(float),
                                    cudaHostAllocDefault));
    FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_cand_),
                                    std::max<size_t>(max_cand, 1) * sizeof(uint32_t),
                                    cudaHostAllocDefault));
    if (g.slot_mode) {
      FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_slots_),
                                      std::max<size_t>(max_cand, 1) * sizeof(uint32_t),
                                      cudaHostAllocDefault));
    }
    FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids_),
                                    std::max<size_t>(max_topn, 1) * sizeof(uint32_t),
                                    cudaHostAllocDefault));
    FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_dists_),
                                    std::max<size_t>(max_topn, 1) * sizeof(float),
                                    cudaHostAllocDefault));
    FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_keys_),
                                    std::max<size_t>(max_topn, 1) * sizeof(unsigned long long),
                                    cudaHostAllocDefault));
    FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_counter_), sizeof(uint32_t),
                                    cudaHostAllocDefault));
  }

  ~GpuFilterWorker() override {
    cudaStreamDestroy(stream_);
    cudaFreeHost(h_query_);
    cudaFreeHost(h_cand_);
    if (h_slots_) cudaFreeHost(h_slots_);
    cudaFreeHost(h_ids_);
    cudaFreeHost(h_dists_);
    cudaFreeHost(h_keys_);
    cudaFreeHost(h_counter_);
  }

  uint32_t* candidate_buffer() override { return h_cand_; }
  uint32_t* slot_buffer() override { return h_slots_; }

  void BeginQuery(const float* query) override {
    FUSION_CUDA_CHECK(cudaSetDevice(g_.device));
    std::memcpy(h_query_, query, g_.dim * sizeof(float));
    FUSION_CUDA_CHECK(cudaMemcpyAsync(d_query_, h_query_, g_.dim * sizeof(float),
                                      cudaMemcpyHostToDevice, stream_));
    LutKernel<<<g_.m, kPQKsub, 0, stream_>>>(d_query_, g_.d_codebook, g_.dsub, d_lut_);
    FUSION_CUDA_CHECK(cudaGetLastError());
  }

  uint32_t Filter(uint32_t n_cand, uint32_t topn, uint32_t* n_unique) override {
    FUSION_CHECK(n_cand <= max_cand_, "too many candidates (%u > %u)", n_cand, max_cand_);
    FUSION_CHECK(topn <= max_topn_, "topn %u exceeds max_topn %u", topn, max_topn_);
    if (n_cand == 0) {
      FUSION_CUDA_CHECK(cudaStreamSynchronize(stream_));
      *n_unique = 0;
      return 0;
    }
    const uint32_t tsize = NextPow2(std::max<uint32_t>(2 * n_cand, 16));
    FUSION_CUDA_CHECK(cudaMemcpyAsync(d_cand_, h_cand_, n_cand * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream_));
    const uint32_t* slots = nullptr;
    if (g_.slot_mode) {
      FUSION_CUDA_CHECK(cudaMemcpyAsync(d_slots_, h_slots_, n_cand * sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, stream_));
      slots = d_slots_;
    }
    FUSION_CUDA_CHECK(cudaMemsetAsync(d_table_, 0xFF, tsize * sizeof(uint32_t), stream_));
    FUSION_CUDA_CHECK(cudaMemsetAsync(d_counter_, 0, sizeof(uint32_t), stream_));

    const uint32_t blocks = std::min<uint32_t>(static_cast<uint32_t>(CeilDiv(n_cand, kBlock)),
                                               static_cast<uint32_t>(4 * g_.num_sms));
    if (g_.m <= kMaxSharedLutM) {
      DedupAdcKernel<true><<<blocks, kBlock, g_.m * kPQKsub * sizeof(float), stream_>>>(
          d_cand_, slots, n_cand, g_.d_codes, g_.m, d_lut_, d_table_, tsize - 1, d_keys_,
          d_counter_);
    } else {
      DedupAdcKernel<false><<<blocks, kBlock, 0, stream_>>>(d_cand_, slots, n_cand, g_.d_codes,
                                                            g_.m, d_lut_, d_table_, tsize - 1,
                                                            d_keys_, d_counter_);
    }
    FUSION_CUDA_CHECK(cudaGetLastError());

    size_t temp = sort_bytes_;
    FUSION_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_sort_temp_, temp, d_keys_, d_sorted_keys_,
                                                     static_cast<int>(n_cand), 0, 64, stream_));
    const uint32_t out = std::min(topn, n_cand);
    FUSION_CUDA_CHECK(cudaMemcpyAsync(h_keys_, d_sorted_keys_, out * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost, stream_));
    FUSION_CUDA_CHECK(cudaMemcpyAsync(h_counter_, d_counter_, sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, stream_));
    FUSION_CUDA_CHECK(cudaStreamSynchronize(stream_));
    *n_unique = *h_counter_;
    const uint32_t res = std::min(out, *h_counter_);
    for (uint32_t i = 0; i < res; ++i) {
      const uint32_t bits = static_cast<uint32_t>(h_keys_[i] >> 32);
      std::memcpy(&h_dists_[i], &bits, sizeof(float));
      h_ids_[i] = static_cast<uint32_t>(h_keys_[i] & 0xFFFFFFFFull);
    }
    return res;
  }

  const uint32_t* result_ids() const override { return h_ids_; }
  const float* result_dists() const override { return h_dists_; }

 private:
  static void Layout(Carver* c, const GpuShared& g, uint32_t max_cand, size_t sort_bytes,
                     GpuFilterWorker* w) {
    const uint32_t tsize = NextPow2(std::max<uint32_t>(2 * max_cand, 16));
    float* q = c->Take<float>(g.dim);
    float* lut = c->Take<float>(static_cast<size_t>(g.m) * kPQKsub);
    uint32_t* cand = c->Take<uint32_t>(max_cand);
    uint32_t* slots = c->Take<uint32_t>(g.slot_mode ? max_cand : 1);
    uint32_t* table = c->Take<uint32_t>(tsize);
    unsigned long long* keys = c->Take<unsigned long long>(max_cand);
    unsigned long long* skeys = c->Take<unsigned long long>(max_cand);
    void* temp = c->Take<char>(sort_bytes);
    uint32_t* counter = c->Take<uint32_t>(1);
    if (w) {
      w->d_query_ = q;
      w->d_lut_ = lut;
      w->d_cand_ = cand;
      w->d_slots_ = slots;
      w->d_table_ = table;
      w->d_keys_ = keys;
      w->d_sorted_keys_ = skeys;
      w->d_sort_temp_ = temp;
      w->d_counter_ = counter;
    }
  }

  GpuShared g_;
  uint32_t max_cand_, max_topn_;
  size_t sort_bytes_;
  cudaStream_t stream_ = nullptr;
  // Device block (owned by the backend's pool).
  float* d_query_ = nullptr;
  float* d_lut_ = nullptr;
  uint32_t* d_cand_ = nullptr;
  uint32_t* d_slots_ = nullptr;  // used in slot mode only
  uint32_t* d_table_ = nullptr;
  unsigned long long* d_keys_ = nullptr;         // (distance bits << 32) | id
  unsigned long long* d_sorted_keys_ = nullptr;
  void* d_sort_temp_ = nullptr;
  uint32_t* d_counter_ = nullptr;
  // Pinned host buffers.
  float* h_query_ = nullptr;
  uint32_t* h_cand_ = nullptr;
  uint32_t* h_slots_ = nullptr;  // slot mode only
  uint32_t* h_ids_ = nullptr;
  float* h_dists_ = nullptr;
  unsigned long long* h_keys_ = nullptr;
  uint32_t* h_counter_ = nullptr;
};

class GpuFilterBackend : public FilterBackend {
 public:
  GpuFilterBackend(const PQCodebook& cb, const uint8_t* codes, uint64_t n, const FilterConfig& cfg) {
    g_.device = cfg.gpu_device;
    FUSION_CUDA_CHECK(cudaSetDevice(g_.device));
    cudaDeviceProp prop;
    FUSION_CUDA_CHECK(cudaGetDeviceProperties(&prop, g_.device));
    g_.num_sms = prop.multiProcessorCount;
    g_.dim = cb.dim();
    g_.m = cb.m();
    g_.dsub = cb.dsub();
    g_.slot_mode = cfg.slot_addressed;

    // Tier: PQ codes resident in device memory for the lifetime of the engine -- all of them
    // (ID addressing), or one slot per code of the node's budget (slot addressing; StoreCodes
    // fills slots as partitions arrive).
    code_bytes_ = n * cb.m();
    FUSION_CUDA_CHECK(cudaMalloc(&d_codes_, std::max<uint64_t>(code_bytes_, 1)));
    if (codes != nullptr) {
      FUSION_CUDA_CHECK(cudaMemcpy(d_codes_, codes, code_bytes_, cudaMemcpyHostToDevice));
    } else {
      FUSION_CHECK(cfg.slot_addressed, "the GPU filter needs the PQ codes");
      FUSION_CUDA_CHECK(cudaMemset(d_codes_, 0, std::max<uint64_t>(code_bytes_, 1)));
    }
    const size_t cb_bytes = static_cast<size_t>(cb.m()) * kPQKsub * cb.dsub() * sizeof(float);
    FUSION_CUDA_CHECK(cudaMalloc(&d_codebook_, cb_bytes));
    FUSION_CUDA_CHECK(cudaMemcpy(d_codebook_, cb.centroids(), cb_bytes, cudaMemcpyHostToDevice));
    g_.d_codes = static_cast<const uint8_t*>(d_codes_);
    g_.d_codebook = static_cast<const float*>(d_codebook_);

    // Contention-free memory pool: one fixed block per worker.
    const uint32_t max_cand = std::max<uint32_t>(cfg.max_candidates, 1);
    size_t sort_bytes = 0;
    FUSION_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, static_cast<const unsigned long long*>(nullptr),
        static_cast<unsigned long long*>(nullptr), static_cast<int>(max_cand), 0, 64));
    const size_t block = GpuFilterWorker::BlockBytes(g_, max_cand, sort_bytes);
    size_t free_b = 0, total_b = 0;
    FUSION_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t need = block * static_cast<size_t>(cfg.num_workers);
    FUSION_CHECK(need <= free_b, "GPU memory: %d workers need %.1f MB of working memory but only "
                 "%.1f MB are free after loading %.1f MB of PQ codes",
                 cfg.num_workers, need / 1e6, free_b / 1e6, code_bytes_ / 1e6);
    FUSION_CUDA_CHECK(cudaMalloc(&pool_, need));
    if (g_.slot_mode) {  // staging for StoreCodes
      FUSION_CUDA_CHECK(cudaStreamCreateWithFlags(&store_stream_, cudaStreamNonBlocking));
      FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_stage_),
                                      static_cast<size_t>(kStageCodes) * g_.m, cudaHostAllocDefault));
      FUSION_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_stage_slots_),
                                      kStageCodes * sizeof(uint32_t), cudaHostAllocDefault));
      FUSION_CUDA_CHECK(cudaMalloc(&d_stage_, static_cast<size_t>(kStageCodes) * g_.m));
      FUSION_CUDA_CHECK(cudaMalloc(&d_stage_slots_, kStageCodes * sizeof(uint32_t)));
    }
    for (int i = 0; i < cfg.num_workers; ++i) {
      workers_.push_back(std::make_unique<GpuFilterWorker>(
          g_, static_cast<char*>(pool_) + static_cast<size_t>(i) * block, max_cand,
          std::max<uint32_t>(cfg.max_topn, 1), sort_bytes));
    }
    Log("GPU backend on %s: %.1f MB for PQ codes (%s), %d x %.2f MB worker blocks", prop.name,
        code_bytes_ / 1e6, g_.slot_mode ? "slots, filled as partitions arrive" : "all resident",
        cfg.num_workers, block / 1e6);
  }

  void StoreCodes(const uint32_t* slots, uint32_t n, const uint8_t* host_codes) override {
    FUSION_CHECK(g_.slot_mode, "StoreCodes on a filter addressed by vector ID");
    std::lock_guard<std::mutex> l(store_mu_);
    FUSION_CUDA_CHECK(cudaSetDevice(g_.device));
    const uint32_t m = g_.m;
    const uint64_t num_slots = code_bytes_ / m;
    // Per batch: gather the codes into pinned memory, copy them and their slots on a dedicated
    // stream, scatter them on the device, and wait. Waiting matters twice: the staging buffers
    // are reused, and queries (on the workers' non-blocking streams, which are not ordered
    // after this one) may read the slots as soon as the partition that names them is loaded.
    for (uint32_t first = 0; first < n; first += kStageCodes) {
      const uint32_t cnt = std::min(kStageCodes, n - first);
      for (uint32_t k = 0; k < cnt; ++k) {
        const uint32_t s = slots[first + k];
        FUSION_CHECK(s < num_slots, "PQ slot %u out of range", s);
        h_stage_slots_[k] = s;
        std::memcpy(h_stage_ + static_cast<size_t>(k) * m, host_codes + static_cast<uint64_t>(s) * m, m);
      }
      FUSION_CUDA_CHECK(cudaMemcpyAsync(d_stage_, h_stage_, static_cast<size_t>(cnt) * m,
                                        cudaMemcpyHostToDevice, store_stream_));
      FUSION_CUDA_CHECK(cudaMemcpyAsync(d_stage_slots_, h_stage_slots_, cnt * sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, store_stream_));
      const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(
          CeilDiv(static_cast<uint64_t>(cnt) * m, kBlock), static_cast<uint64_t>(4 * g_.num_sms)));
      ScatterCodesKernel<<<blocks, kBlock, 0, store_stream_>>>(
          static_cast<const uint8_t*>(d_stage_), static_cast<const uint32_t*>(d_stage_slots_), cnt, m,
          static_cast<uint8_t*>(d_codes_));
      FUSION_CUDA_CHECK(cudaGetLastError());
      FUSION_CUDA_CHECK(cudaStreamSynchronize(store_stream_));
    }
  }

  ~GpuFilterBackend() override {
    workers_.clear();
    if (store_stream_) cudaStreamDestroy(store_stream_);
    if (h_stage_) cudaFreeHost(h_stage_);
    if (h_stage_slots_) cudaFreeHost(h_stage_slots_);
    if (d_stage_) cudaFree(d_stage_);
    if (d_stage_slots_) cudaFree(d_stage_slots_);
    cudaFree(pool_);
    cudaFree(d_codebook_);
    cudaFree(d_codes_);
  }

  const char* name() const override { return "gpu"; }
  FilterWorker* worker(int i) override { return workers_.at(static_cast<size_t>(i)).get(); }
  uint64_t code_bytes() const override { return code_bytes_; }

 private:
  GpuShared g_;
  uint64_t code_bytes_ = 0;
  void* d_codes_ = nullptr;
  void* d_codebook_ = nullptr;
  void* pool_ = nullptr;
  // StoreCodes staging (slot mode only).
  std::mutex store_mu_;
  cudaStream_t store_stream_ = nullptr;
  uint8_t* h_stage_ = nullptr;
  uint32_t* h_stage_slots_ = nullptr;
  void* d_stage_ = nullptr;
  void* d_stage_slots_ = nullptr;
  std::vector<std::unique_ptr<GpuFilterWorker>> workers_;
};

}  // namespace

std::unique_ptr<FilterBackend> CreateGpuFilter(const PQCodebook& cb, const uint8_t* codes,
                                               uint64_t n, const FilterConfig& cfg) {
  return std::make_unique<GpuFilterBackend>(cb, codes, n, cfg);
}

bool GpuFilterAvailable() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace fusion
