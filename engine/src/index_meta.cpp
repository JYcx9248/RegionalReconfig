#include "fusion/index_meta.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

namespace fusion {

std::string IndexMeta::Get(const std::string& key) const {
  auto it = kv_.find(key);
  FUSION_CHECK(it != kv_.end(), "index metadata is missing key '%s'", key.c_str());
  return it->second;
}

std::string IndexMeta::GetOr(const std::string& key, const std::string& def) const {
  auto it = kv_.find(key);
  return it == kv_.end() ? def : it->second;
}

uint64_t IndexMeta::GetU64(const std::string& key) const { return std::stoull(Get(key)); }
double IndexMeta::GetDouble(const std::string& key) const { return std::stod(Get(key)); }

void IndexMeta::Save(const std::string& path) const {
  std::ofstream out(path);
  FUSION_CHECK(out.good(), "cannot write %s", path.c_str());
  for (const auto& kv : kv_) out << kv.first << "=" << kv.second << "\n";
  FUSION_CHECK(out.good(), "write failed for %s", path.c_str());
}

IndexMeta IndexMeta::Load(const std::string& path) {
  std::ifstream in(path);
  FUSION_CHECK(in.good(), "cannot read %s (is this an index directory?)", path.c_str());
  IndexMeta m;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    m.kv_[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return m;
}

void PostingLists::Save(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t hdr[2] = {num_lists(), ids.size()};
  AppendToFile(f, hdr, sizeof(hdr), path);
  WriteVector(f, offsets, path);
  WriteVector(f, ids, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

PostingLists PostingLists::Load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t hdr[2];
  FUSION_CHECK(std::fread(hdr, sizeof(hdr), 1, f) == 1, "bad header in %s", path.c_str());
  PostingLists pl;
  pl.offsets.resize(hdr[0] + 1);
  pl.ids.resize(hdr[1]);
  bool ok = std::fread(pl.offsets.data(), sizeof(uint64_t), pl.offsets.size(), f) ==
                pl.offsets.size() &&
            (pl.ids.empty() ||
             std::fread(pl.ids.data(), sizeof(uint32_t), pl.ids.size(), f) == pl.ids.size());
  std::fclose(f);
  FUSION_CHECK(ok, "truncated posting lists %s", path.c_str());
  FUSION_CHECK(pl.offsets.back() == pl.ids.size(), "corrupt posting lists %s", path.c_str());
  return pl;
}

void SaveU32Array(const std::string& path, const std::vector<uint32_t>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t n = v.size();
  AppendToFile(f, &n, sizeof(n), path);
  WriteVector(f, v, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

std::vector<uint32_t> LoadU32Array(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t n = 0;
  FUSION_CHECK(std::fread(&n, sizeof(n), 1, f) == 1, "bad header in %s", path.c_str());
  std::vector<uint32_t> v(n);
  bool ok = n == 0 || std::fread(v.data(), sizeof(uint32_t), n, f) == n;
  std::fclose(f);
  FUSION_CHECK(ok, "truncated array %s", path.c_str());
  return v;
}

}  // namespace fusion
