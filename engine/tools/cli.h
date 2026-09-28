// Minimal "--key value" / "--flag" command-line parsing for the tools.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "fusion/common.h"

namespace fusion {

class Args {
 public:
  Args(int argc, char** argv, const std::set<std::string>& flags) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + a);
      std::string key = a.substr(2);
      size_t eq = key.find('=');
      if (eq != std::string::npos) {
        kv_[key.substr(0, eq)] = key.substr(eq + 1);
      } else if (flags.count(key)) {
        kv_[key] = "1";
      } else {
        if (i + 1 >= argc) throw std::runtime_error("missing value for --" + key);
        kv_[key] = argv[++i];
      }
    }
  }

  bool Has(const std::string& k) const { return kv_.count(k) != 0; }
  bool Flag(const std::string& k) const {
    used_.insert(k);
    return Has(k) && kv_.at(k) != "0";
  }
  std::string Str(const std::string& k, const std::string& def = "") const {
    used_.insert(k);
    return Has(k) ? kv_.at(k) : def;
  }
  std::string Required(const std::string& k) const {
    if (!Has(k)) throw std::runtime_error("missing required option --" + k);
    return Str(k);
  }
  uint64_t U64(const std::string& k, uint64_t def) const {
    return Has(k) ? std::stoull(Str(k)) : def;
  }
  double F64(const std::string& k, double def) const { return Has(k) ? std::stod(Str(k)) : def; }
  std::vector<uint64_t> List(const std::string& k, const std::string& def) const {
    return ParseUintList(Str(k, def));
  }
  // Warns about options that were given but never read (likely typos).
  void WarnUnused(const std::set<std::string>& flags) const {
    for (const auto& kv : kv_) {
      if (!used_.count(kv.first) && !flags.count(kv.first))
        std::fprintf(stderr, "warning: unknown option --%s ignored\n", kv.first.c_str());
    }
  }

 private:
  std::map<std::string, std::string> kv_;
  mutable std::set<std::string> used_;
};

}  // namespace fusion
