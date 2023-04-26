#ifndef EXPERIMENT_HPP_
#define EXPERIMENT_HPP_

#include <map>
#include <string>
#include <vector>
#include <array>
#include <inttypes.h>
#include <getopt.h>

#include "Types.hpp"
#include "FPGrowth/fptree.hpp"

/* FPGrowth */
SetPattern findFrequentPattern(const CodebookType &codebook, uint64_t minimum_support_threshold, const std::vector<int> &centroidsNum) {
  const FPTree fptree{ codebook, minimum_support_threshold, centroidsNum};
  return fptree_growth(fptree);
}

/* Sorting Utils for KNN */
template<typename T>
static inline void maybeInsertNeighbor(std::vector<IdxDistPairBase<T>> &neighbors_bsf, IdxDistPairBase<T> newNeighbor) {
  size_t len = neighbors_bsf.size();
  size_t i = len - 1;
  auto dist = newNeighbor.dist;

  if (dist < neighbors_bsf[i].dist) {
    neighbors_bsf[i] = newNeighbor;
  }
  while (i > 0 && neighbors_bsf[i-1].dist > dist) {
    // swap new and previous neighbor
    IdxDistPairBase<T> tmp = neighbors_bsf[i-1];
    neighbors_bsf[i-1] = neighbors_bsf[i];
    neighbors_bsf[i] = tmp;
    i--;
  }
}

template<typename T>
static inline std::vector<IdxDistPairBase<T>> KNNFromDists(const T* dists, int len, int k) {
  std::vector<IdxDistPairBase<T>> ret(k);
  for (int i = 0; i < k; i++) {
    ret[i] = IdxDistPairBase<T>{i, dists[i]};
  }
  
  std::sort(ret.begin(), ret.end(), 
    [](const IdxDistPairBase<T>& a, const IdxDistPairBase<T>& b) -> bool {
      return a.dist < b.dist;
    }
  );

  for (int i = k; i < len; i++) {
    maybeInsertNeighbor(ret, IdxDistPairBase<T>{i, dists[i]});
  }
  return ret;
}

template<typename T>
static inline void KNNFromDists(const T* distsvec, int len, int k, f::float_maxheap_t &res, int res_offset) {
  std::vector<IdxDistPairBase<T>> temppairs(k);
  for (int i = 0; i < k; i++) {
    temppairs[i] = IdxDistPairBase<T>{i, distsvec[i]};
  }
  std::sort(temppairs.begin(), temppairs.end(), 
    [](const IdxDistPairBase<T>& a, const IdxDistPairBase<T>& b) -> bool {
      return a.dist < b.dist;
    }
  );

  float * res_dist = res.val + res_offset;
  int * res_label = res.ids + res_offset;
  for (int i = 0; i < k; i++) {
    *res_dist = temppairs[i].dist;
    *res_label = temppairs[i].idx;
  }

  for (int i = k; i < len; i++) {
    // maybe insert neighbor
    size_t idx = k - 1;
    auto newneighbordist = distsvec[i];

    if (newneighbordist < res_dist[idx]) {
      res_dist[idx] = distsvec[i];
      res_label[idx] = i;
    }
    while (idx > 0 && res_dist[idx-1] > newneighbordist) {
      // swap new and previous neighbor
      std::swap(res_dist[idx-1], res_dist[idx]);
      std::swap(res_label[idx-1], res_label[idx]);
      idx--;
    }
  }
}

/* Argument Parsing Utils */
class ArgsParse {
public:
  struct opt {
    std::string str;
    char type;
    std::string val;
  };
  std::map<std::string, char> optType;

  ArgsParse(int argc, char **argv, std::vector<ArgsParse::opt> long_options_raw, std::string helpMsg="") {
    const int options_len = long_options_raw.size();
    
    std::map<char, std::string> optChrToKey;
    
    std::vector<struct option> long_options(options_len + 1);
    char opt_chr = 'a';
    for (int i=0; i<options_len; i++) {
      ArgsParse::opt &opt = long_options_raw[i];
      long_options[i] = { opt.str.c_str(), required_argument, 0, opt_chr };

      switch (opt.type) {
        case 's': strMap.insert({ opt.str, opt.val }); break;
        case 'i': intMap.insert({ opt.str, std::stoi(opt.val) }); break;
        case 'f': floatMap.insert({ opt.str, std::stof(opt.val) }); break;
        case 'b': boolMap.insert({ opt.str, (bool)std::stoi(opt.val) }); break;
      }

      optType.insert({ opt.str, opt.type });
      optChrToKey.insert({ opt_chr, opt.str });
      opt_chr++;
    }
    long_options[options_len] = { 0, 0, 0, 0 };

    while (true) {
      int option_index = 0;
      char c = getopt_long(argc, argv, "", (struct option*)long_options.data(), &option_index);
      if (c == '?') {
        std::cout << helpMsg << std::endl;
        exit(0);
      }
      if (c == -1) break;
      std::string key = optChrToKey.find(c)->second;
      switch (optType.find(key)->second) {
        case 's': strMap.at(key) = optarg; break;
        case 'i': intMap.at(key) = std::stoi(optarg); break;
        case 'f': floatMap.at(key) = std::stof(optarg); break;
        case 'b': boolMap.at(key) = (bool)std::stoi(optarg); break;
      }
    }
  }

  void printArgs() const {
    std::cout << "Arguments Passed:" << std::endl;
    for (auto const& it: optType) {
      std::cout << "\t" << it.first << " = ";

      switch (it.second) {
        case 's': std::cout << strMap.at(it.first) << std::endl; break;
        case 'i': std::cout << intMap.at(it.first) << std::endl; break;
        case 'f': std::cout << floatMap.at(it.first) << std::endl; break;
        case 'b': std::cout << boolMap.at(it.first) << std::endl; break;
      }
    }
  }

  template<typename T=std::string>
  inline T at(std::string key) const {
    return strMap.at(key);
  }

  template<typename T=std::string>
  inline T operator[](std::string key) const {}

private:
  std::map<std::string, std::string> strMap;
  std::map<std::string, int> intMap;
  std::map<std::string, float> floatMap;
  std::map<std::string, bool> boolMap;
};

/* Template specialization for ArgsParse class */
template<>
inline int ArgsParse::at<int>(std::string key) const {
  return intMap.at(key);
}

template<>
inline float ArgsParse::at<float>(std::string key) const {
  return floatMap.at(key);
}

template<>
inline bool ArgsParse::at<bool>(std::string key) const {
  return boolMap.at(key);
}

template<>
inline std::string ArgsParse::operator[]<std::string>(std::string key) const {
  return strMap.at(key);
}

template<>
inline int ArgsParse::operator[]<int>(std::string key) const {
  return intMap.at(key);
}

template<>
inline float ArgsParse::operator[]<float>(std::string key) const {
  return floatMap.at(key);
}

template<>
inline bool ArgsParse::operator[]<bool>(std::string key) const {
  return boolMap.at(key);
}

/* VAQ Hardcoding Utils */
std::vector<int> parseVAQHardcode(std::string str) {
  std::vector<int> ret;
  ret.reserve(64);

  std::vector<std::string> parsed;
  std::stringstream ss(str);
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    parsed.push_back(substr);
  }

  for (std::string token: parsed) {
    ret.push_back(std::stoi(token));
  }

  return ret;
}

/* Measuring Accuracy Utils */
template <int IdxOffset=0, class T>
double getAvgRecall(const T& pairs, const std::vector<std::vector<int>> &topnn, const int K) {
  double ans = 0.0;

  for (int p_idx=0; p_idx<(int)pairs.size(); p_idx++) {
    int ct = 0;
    for (auto p: pairs[p_idx]) {
      for (int j=0; j<K; j++) {
        if (p.idx == topnn[p_idx][j]-IdxOffset) {
          ct++;
          break;
        }
      }
    }
    ans += ((double)ct) / K;
  }
  ans /= pairs.size();
  return ans;
}
template <int IdxOffset=0>
double getAvgRecall(const std::vector<int> &labels, const std::vector<std::vector<int>> &topnn, const int K) {
  double ans = 0.0;

  int nq = labels.size() / K;
  for (int q_idx=0; q_idx<nq; q_idx++) {
    int ct = 0;
    for (int k_idx=0; k_idx<K; k_idx++) {
      for (int j=0; j<K; j++) {
        if (labels[q_idx * K + k_idx] == topnn[q_idx][j]-IdxOffset) {
          ct++;
          break;
        }
      }
    }
    ans += ((double)ct) / K;
  }
  ans /= nq;
  return ans;
}

template <int IdxOffset=0, class T>
double getRecallAtR(const T& pairs, const std::vector<std::vector<int>> &topnn) {
  double ans = 0.0;
  for (int p_idx=0; p_idx<(int)pairs.size(); p_idx++) {
    int truenn = topnn[p_idx][0];
    for (auto p: pairs[p_idx]) {
      if (truenn-IdxOffset == p.idx) {
        ans += 1;
        break;
      }
    }
  }
  ans /= pairs.size();
  return ans;
}
template <int IdxOffset=0>
double getRecallAtR(const std::vector<int> &labels, const std::vector<std::vector<int>> &topnn, const int K) {
  int nq = labels.size() / K;
  double ans = 0.0;
  for (int p_idx=0; p_idx<nq; p_idx++) {
    int truenn = topnn[p_idx][0];
    for (int k_idx=0; k_idx<K; k_idx++) {
      if (truenn-IdxOffset == labels[p_idx * K + k_idx]) {
        ans += 1;
        break;
      }
    }
  }
  ans /= nq;
  return ans;
}

template <int IdxOffset=0, class T>
double getMeanAveragePrecision(const T& pairs, const std::vector<std::vector<int>> &topnn, const int K) {
  double ans = 0.0;

  for (int p_idx=0; p_idx<(int)pairs.size(); p_idx++) {
    double ap = 0;
    for (int r=1; r<=K; r++) {
      bool isR_kExact = false;
      for (int j=0; j<K; j++) {
        if (pairs[p_idx][r-1].idx == topnn[p_idx][j]-IdxOffset) {
          isR_kExact = true;
          break;
        }
      }
      if (isR_kExact) {
        int ct = 0;
        for (int j=0; j<r; j++) {
          for (int jj=0; jj<r; jj++) {
            if (pairs[p_idx][j].idx == topnn[p_idx][jj]-IdxOffset) {
              ct++;
              break;
            }
          }
        }
        ap += (double)ct/r;
      }
    }
    ans += ap / K;
  }
  ans /= pairs.size();
  return ans;
}
template <int IdxOffset=0>
double getMeanAveragePrecision(const std::vector<int>& labels, const std::vector<std::vector<int>> &topnn, const int K) {
  double ans = 0.0;

  const int nq = labels.size() / K;
  for (int p_idx=0; p_idx<nq; p_idx++) {
    double ap = 0;
    for (int r=1; r<=K; r++) {
      bool isR_kExact = false;
      for (int j=0; j<K; j++) {
        if (labels[p_idx * K + (r - 1)] == topnn[p_idx][j]-IdxOffset) {
          isR_kExact = true;
          break;
        }
      }
      if (isR_kExact) {
        int ct = 0;
        for (int j=0; j<r; j++) {
          for (int jj=0; jj<r; jj++) {
            if (labels[p_idx * K + j] == topnn[p_idx][jj]-IdxOffset) {
              ct++;
              break;
            }
          }
        }
        ap += (double)ct/r;
      }
    }
    ans += ap / K;
  }
  ans /= nq;
  return ans;
}

/* Clustering functions for BitVecEngine */
void computeClusterIndex(const Matrixf &dataset, const Matrixf& centroids, std::vector<int> &clusterIdx) {
  for (auto row: dataset) {
    int nearestIdx = -1;
    float nearestDist = std::numeric_limits<float>::max();
    int c_idx = 0;
    for (auto c_row: centroids) {
      float dist = 0;
      int dimen = row.size();
      for (int i=0; i<dimen; i++) {
        dist += (row[i]-c_row[i])*(row[i]-c_row[i]);
      }

      if (dist < nearestDist) {
        nearestIdx = c_idx;
        nearestDist = dist;
      }
      c_idx++;
    }
    clusterIdx.push_back(nearestIdx);
  }
}

void computeClusterIndexDiskResident(std::string filepath, const Matrixf& centroids, std::vector<int> &clusterIdx, int N, int batch=10000000) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  while (true) {
    Matrixf loadedData;
    int loadedData_iter = 0;
    while (loadedData_iter < batch) {
      std::vector<float> v(N);
      if (fread(v.data(), sizeof(float), N, infile) == 0) {
        break;
      }
      loadedData.push_back(v);
      loadedData_iter++;
    }

    if (loadedData_iter == 0) {
      break;
    }

    for (auto row: loadedData) {
      int nearestIdx = -1;
      float nearestDist = std::numeric_limits<float>::max();
      int c_idx = 0;
      for (auto c_row: centroids) {
        float dist = 0;
        int dimen = row.size();
        for (int i=0; i<dimen; i++) {
          dist += (row[i]-c_row[i])*(row[i]-c_row[i]);
        }

        if (dist < nearestDist) {
          nearestIdx = c_idx;
          nearestDist = dist;
        }
        c_idx++;
      }
      clusterIdx.push_back(nearestIdx);
    }

    if (loadedData_iter < batch) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

/* Misc Functions */
template<typename T>
void printArr(T * a, size_t n) {
  for (size_t i=0; i<n; i++) {
    std::cout << (*(a + i)) << ", ";
  }
  std::cout << std::endl;
}

inline bool isInRange(int x, int lower, int upper) {
  return (x >= lower && x <= upper);
}

#endif // EXPERIMENT_HPP_