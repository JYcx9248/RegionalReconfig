// On-disk index layout: file names, the metadata file and the posting-list (vector-ID) file.
//
// An index directory contains
//   meta.txt              key=value metadata
//   heads.bin             uint32 data IDs of the posting-list centroids ("heads")
//   centroids.bin         head vectors (bin format, native dtype)
//   graph.hnsw            navigation graph over the heads (hnswlib)
//   postings.bin          vector IDs of every posting list (CSR)       -> host memory tier
//   primary.bin           nearest head of every vector (used by the layout step)
//   pq_codebook.bin       PQ codebook
//   pq_codes.bin          PQ codes of every vector                     -> GPU memory tier
//   layout_<name>.map     vector -> (SSD page, slot) mapping           -> host memory tier
//   vectors_<name>.bin    raw vectors packed into 4 KB pages           -> SSD tier
#pragma once

#include <map>
#include <string>
#include <vector>

#include "fusion/common.h"

namespace fusion {

namespace files {
constexpr const char* kMeta = "meta.txt";
constexpr const char* kHeads = "heads.bin";
constexpr const char* kCentroids = "centroids.bin";
constexpr const char* kGraph = "graph.hnsw";
constexpr const char* kPostings = "postings.bin";
constexpr const char* kPrimary = "primary.bin";
constexpr const char* kPQCodebook = "pq_codebook.bin";
constexpr const char* kPQCodes = "pq_codes.bin";
inline std::string LayoutMap(const std::string& name) { return "layout_" + name + ".map"; }
inline std::string LayoutVectors(const std::string& name) { return "vectors_" + name + ".bin"; }
}  // namespace files

class IndexMeta {
 public:
  void Set(const std::string& key, const std::string& value) { kv_[key] = value; }
  void Set(const std::string& key, const char* value) { kv_[key] = value; }
  template <class T>
  void Set(const std::string& key, T value) {
    kv_[key] = std::to_string(value);
  }
  bool Has(const std::string& key) const { return kv_.count(key) != 0; }
  std::string Get(const std::string& key) const;
  std::string GetOr(const std::string& key, const std::string& def) const;
  uint64_t GetU64(const std::string& key) const;
  double GetDouble(const std::string& key) const;

  void Save(const std::string& path) const;
  static IndexMeta Load(const std::string& path);
  const std::map<std::string, std::string>& items() const { return kv_; }

 private:
  std::map<std::string, std::string> kv_;
};

// Posting lists stored as CSR: ids[offsets[c] .. offsets[c+1]) are the vector IDs of list c.
// Only IDs are kept (no vector content) -- the multi-tiered index's host-memory tier.
struct PostingLists {
  std::vector<uint64_t> offsets;  // num_lists + 1
  std::vector<uint32_t> ids;

  uint32_t num_lists() const {
    return offsets.empty() ? 0 : static_cast<uint32_t>(offsets.size() - 1);
  }
  uint32_t Size(uint32_t c) const { return static_cast<uint32_t>(offsets[c + 1] - offsets[c]); }
  const uint32_t* List(uint32_t c) const { return ids.data() + offsets[c]; }

  void Save(const std::string& path) const;
  static PostingLists Load(const std::string& path);
};

// Plain uint32 arrays (heads.bin, primary.bin): uint64 count, then values.
void SaveU32Array(const std::string& path, const std::vector<uint32_t>& v);
std::vector<uint32_t> LoadU32Array(const std::string& path);

}  // namespace fusion
