#include "fusion/kmeans.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

namespace fusion {

std::vector<float> KMeansTrain(const float* x, size_t n, size_t d, const KMeansParams& p) {
  FUSION_CHECK(n > 0 && d > 0, "k-means on empty data");
  const uint32_t k = static_cast<uint32_t>(std::min<size_t>(p.k, n));
  std::mt19937_64 rng(p.seed);

  std::vector<float> cent(static_cast<size_t>(k) * d);
  if (k == n) {
    std::memcpy(cent.data(), x, n * d * sizeof(float));
    return cent;
  }
  // k-means++ seeding: each new center is drawn with probability proportional to the squared
  // distance to the nearest center chosen so far. Costs about one Lloyd iteration.
  {
    std::vector<float> mind(n, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> first(0, n - 1);
    size_t chosen = first(rng);
    for (uint32_t c = 0; c < k; ++c) {
      std::memcpy(&cent[static_cast<size_t>(c) * d], x + chosen * d, d * sizeof(float));
      if (c + 1 == k) break;
      const float* cc = &cent[static_cast<size_t>(c) * d];
      double total = 0;
#pragma omp parallel for schedule(static) reduction(+ : total)
      for (size_t i = 0; i < n; ++i) {
        mind[i] = std::min(mind[i], L2Sqr(x + i * d, cc, d));
        total += mind[i];
      }
      if (total <= 0) {  // all remaining points coincide with centers: pick any
        chosen = first(rng);
        continue;
      }
      std::uniform_real_distribution<double> u(0.0, total);
      double r = u(rng), acc = 0;
      chosen = n - 1;
      for (size_t i = 0; i < n; ++i) {
        acc += mind[i];
        if (acc >= r) {
          chosen = i;
          break;
        }
      }
    }
  }

  std::vector<uint32_t> assign(n);
  std::vector<float> err(n);
  std::vector<float> penalty(k, 0.f);
  std::vector<uint64_t> counts(k);
  const int nthreads = omp_in_parallel() ? 1 : omp_get_max_threads();
  std::vector<double> sums_all(static_cast<size_t>(nthreads) * k * d);
  std::vector<uint64_t> counts_all(static_cast<size_t>(nthreads) * k);

  for (uint32_t it = 0; it < p.iters; ++it) {
    // Assignment step.
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (size_t i = 0; i < n; ++i) {
      const float* xi = x + i * d;
      uint32_t best = 0;
      float best_d = L2Sqr(xi, cent.data(), d);
      float best_cost = best_d + penalty[0];
      for (uint32_t c = 1; c < k; ++c) {
        float dd = L2Sqr(xi, cent.data() + static_cast<size_t>(c) * d, d);
        float cost = dd + penalty[c];
        if (cost < best_cost) {
          best_cost = cost;
          best_d = dd;
          best = c;
        }
      }
      assign[i] = best;
      err[i] = best_d;
    }

    // Update step with per-thread accumulators.
    std::fill(sums_all.begin(), sums_all.end(), 0.0);
    std::fill(counts_all.begin(), counts_all.end(), 0);
#pragma omp parallel num_threads(nthreads)
    {
      int t = omp_get_thread_num();
      double* sums = &sums_all[static_cast<size_t>(t) * k * d];
      uint64_t* cnt = &counts_all[static_cast<size_t>(t) * k];
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; ++i) {
        uint32_t c = assign[i];
        cnt[c]++;
        double* s = sums + static_cast<size_t>(c) * d;
        const float* xi = x + i * d;
        for (size_t j = 0; j < d; ++j) s[j] += xi[j];
      }
    }
    std::fill(counts.begin(), counts.end(), 0);
    for (int t = 1; t < nthreads; ++t) {
      for (size_t j = 0; j < static_cast<size_t>(k) * d; ++j)
        sums_all[j] += sums_all[static_cast<size_t>(t) * k * d + j];
    }
    for (int t = 0; t < nthreads; ++t)
      for (uint32_t c = 0; c < k; ++c) counts[c] += counts_all[static_cast<size_t>(t) * k + c];
    for (uint32_t c = 0; c < k; ++c) {
      if (counts[c] == 0) continue;
      double inv = 1.0 / static_cast<double>(counts[c]);
      for (size_t j = 0; j < d; ++j)
        cent[static_cast<size_t>(c) * d + j] = static_cast<float>(sums_all[c * d + j] * inv);
    }

    // Split a large cluster into every empty one (as Faiss does).
    constexpr float kEps = 1.f / 1024.f;
    for (uint32_t ci = 0; ci < k; ++ci) {
      if (counts[ci] != 0) continue;
      uint32_t cj = 0;
      for (int tries = 0;; ++tries) {
        std::uniform_int_distribution<uint64_t> pick(0, n - 1);
        uint64_t r = pick(rng);
        uint64_t acc = 0;
        for (cj = 0; cj < k; ++cj) {
          acc += counts[cj];
          if (acc > r) break;
        }
        if (cj < k && counts[cj] > 1) break;
        if (tries > 1000) {  // fall back to the largest cluster
          cj = static_cast<uint32_t>(std::max_element(counts.begin(), counts.end()) -
                                     counts.begin());
          break;
        }
      }
      float* a = &cent[static_cast<size_t>(ci) * d];
      float* b = &cent[static_cast<size_t>(cj) * d];
      std::memcpy(a, b, d * sizeof(float));
      for (size_t j = 0; j < d; ++j) {
        float delta = kEps * (std::fabs(b[j]) + 1e-3f);
        if (j % 2 == 0) {
          a[j] += delta;
          b[j] -= delta;
        } else {
          a[j] -= delta;
          b[j] += delta;
        }
      }
      counts[ci] = counts[cj] / 2;
      counts[cj] -= counts[ci];
    }

    // Balance penalty for the next iteration.
    if (p.balance > 0.f) {
      double mean_err = std::accumulate(err.begin(), err.end(), 0.0) / static_cast<double>(n);
      double avg_size = static_cast<double>(n) / k;
      for (uint32_t c = 0; c < k; ++c)
        penalty[c] = static_cast<float>(p.balance * mean_err * counts[c] / avg_size);
    }
  }
  return cent;
}

}  // namespace fusion
