#include <algorithm>
#include <omp.h>
#include <cassert>
#include <tuple>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cstdio>

#include "BitVecEngine.hpp"

void BitVecEngine::loadBitV(bitvectors bitVectors) {
  data = bitVectors;
  
  // psychically store each subvector of data separatelly for progressive query filtering
  for (const bitv& datum: data) {
    for (int i=0; i<SubVector; i++) {
      const int minIter = (i * SubVectorLen);

      bitv v(actualBitVLen(N/SubVector), 0);
      for (int j=0; j<SubVectorLen; ++j) {
        v[j] = datum[j+minIter];
      }
      dataSplitted[i].push_back(v);
    }
  }
}

void BitVecEngine::loadOriginal(Matrixf oriDataset) {
  originalData = oriDataset;
}

void BitVecEngine::loadCentroidsBin(bitvectors _centroids) {
  this->centroidsBin = _centroids;
}

void BitVecEngine::loadCentroids(Matrixf _centroids) {
  this->centroids = _centroids;
  this->centroids_size = this->centroids.size();
}



void BitVecEngine::loadClusterInfo(std::vector<int> clustIdx, int _centroids_size) {
  this->centroids_size = _centroids_size;
  clusterMembers.resize(centroids_size);
  int dataIdx = 0;
  for (int ci: clustIdx) {
    clusterMembers[ci].push_back(dataIdx);
    dataIdx++;
  }
}

/**
 * Helper Function for Query
 */

/**
 * Simple Query KNN
 */
std::vector<std::vector<IdxDistPair>> query_sort(const bitvectors &data, const bitvectors &queries, const int k) {
  const int dataNRows = data.size();
  std::vector<std::vector<IdxDistPair>> answers(queries.size());

  int q_idx = 0;
  for (const bitv& q: queries) {
    std::vector<uint32_t> pairsDist(dataNRows);
    
    for (int dataIndex = 0; dataIndex < dataNRows; dataIndex++) {
      pairsDist[dataIndex] = hammingDist(q, data[dataIndex]);
    }

    answers[q_idx] = BitVecEngine::KNNFromDists(pairsDist.data(), pairsDist.size(), k);
    q_idx++;
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPair>> query_sort_early_abandon(const bitvectors &data, const bitvectors &queries, const int k) {
  const int dataNRows = data.size();
  std::vector<std::vector<IdxDistPair>> answers(queries.size());
  
  int q_idx = 0;
  for (const bitv& q: queries) {
    std::vector<IdxDistPair> pairs;
    pairs.reserve(k+1);

    auto insertionSort = [&pairs](int idxStart, int idx, uint32_t dist) {
      if (pairs.size() > 0) {
        int i = idxStart-1;
        for (; i>0; i--) {
          if (dist > pairs[i-1].dist) {
            break;
          }
        }
        pairs.emplace(pairs.begin() + i, idx, dist);
      } else {
        pairs.emplace(pairs.begin(), idx, dist);
      }
    };
    
    uint32_t dist, bsfK = 0;
    
    int dataIndex = 0;
    for (; dataIndex < k; dataIndex++) {
      dist = hammingDist(q, data[dataIndex]);
      if (dist > bsfK) {
        bsfK = dist;
      }
      insertionSort(dataIndex, dataIndex, dist);
    }
    
    for (; dataIndex < dataNRows; dataIndex++) {
      dist = hammingDistEarlyAbandon(q, data[dataIndex], bsfK);
      if (dist < bsfK) {
        insertionSort(k, dataIndex, dist);
        pairs.pop_back();
        bsfK = pairs[k-1].dist;
      }
    }

    pairs.shrink_to_fit();
    answers[q_idx] = std::move(pairs);
    q_idx++;
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPair>> query_heap(const bitvectors &data, const bitvectors &queries, const int k) {
  std::vector<std::vector<IdxDistPair>> answers(queries.size());
  auto comparator = [](IdxDistPair const& a, IdxDistPair const& b) -> bool {
    return a.dist < b.dist;
  };

  for (int idx_q=0; idx_q<(int)queries.size(); idx_q++) {
    std::vector<IdxDistPair> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    
    int iter = 0;
    for (const bitv& datum: data) {
      pairs.push_back(IdxDistPair(iter, hammingDist(queries[idx_q], datum)));
      std::push_heap(pairs.begin(), pairs.end(), comparator);
      if (iter+1 > k) {
        std::pop_heap(pairs.begin(), pairs.end(), comparator);
        pairs.pop_back();
      }
      iter++;
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPair>> query_heap_early_abandon(const bitvectors &data, const bitvectors &queries, const int k) {
  const int queryNRows = queries.size();
  std::vector<std::vector<IdxDistPair>> answers(queryNRows);
  auto comparator = [](IdxDistPair const& a, IdxDistPair const& b) -> bool {
    return a.dist < b.dist;
  };

  for (int idx_q=0; idx_q<queryNRows; idx_q++) {
    std::vector<IdxDistPair> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    uint32_t bsfK = 0;
    uint32_t dist;
    
    int iter = 0;
    for (const bitv& datum: data) {
      if (iter < k) {
        dist = hammingDist(queries[idx_q], datum);
        pairs.push_back(IdxDistPair(iter, dist));
        std::push_heap(pairs.begin(), pairs.end(), comparator);
        if (dist > bsfK)  {
          bsfK = dist;
        }
      } else {
        dist = hammingDistEarlyAbandon(queries[idx_q], datum, bsfK);
        if (dist < bsfK) {
          pairs.push_back(IdxDistPair(iter, dist));
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          std::pop_heap(pairs.begin(), pairs.end(), comparator);
          pairs.pop_back();
          bsfK = (pairs.front()).dist;
        }
      }
      iter++;
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }
  
  return answers;
}



/**
 * Query with cluster information
 */
std::vector<std::vector<IdxDistPair>> queryWithClusterInfo_heap(const bitvectors &data, const bitvectors &queries, const Matrixf &oriQueries, const Matrixf &centroids, const Matrixi &clusterMembers, const int k, int atleastCluster) {
  const int centroids_size = centroids.size();
  auto comparator = [](IdxDistPair const& a, IdxDistPair const& b) -> bool {
    return a.dist < b.dist;
  };
  
  std::vector<std::vector<IdxDistPair>> answers(queries.size());
  
  for (int idx_q = 0; idx_q < (int)queries.size(); idx_q++) {
    std::vector<IdxDistPairFloat> nearestCluster(centroids_size);
    
    for (int c_idx=0; c_idx<centroids_size; c_idx++) {
      nearestCluster[c_idx] = IdxDistPairFloat(c_idx, euclideanDistNoSQRT(oriQueries[idx_q], centroids[c_idx]));
    }
    std::sort(nearestCluster.begin(), nearestCluster.end(),
      [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
        return a.dist < b.dist;
      }
    );

    int nearestClusterCumulative = 0;
    int farthestClusterIdx = 0;
    for (const IdxDistPairFloat& nc: nearestCluster) {
      farthestClusterIdx++;
      nearestClusterCumulative += clusterMembers[nc.idx].size();
      if (nearestClusterCumulative >= k) {
        break;
      }
    }
    while (farthestClusterIdx < atleastCluster) {
      nearestClusterCumulative += clusterMembers[nearestCluster[farthestClusterIdx].idx].size();
      farthestClusterIdx++;
    }

    std::vector<IdxDistPair> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    int iter = 0;
    for (int i=0; i<farthestClusterIdx; i++) {
      for (int cm: clusterMembers[nearestCluster[i].idx]) {
        pairs.push_back(IdxDistPair(cm, hammingDist(queries[idx_q], data[cm])));
        std::push_heap(pairs.begin(), pairs.end(), comparator);
        if (iter+1 > k) {
          std::pop_heap(pairs.begin(), pairs.end(), comparator);
          pairs.pop_back();
        }
        iter++;
      }
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }

  return answers;
}

std::vector<std::vector<IdxDistPair>> queryWithClusterInfo_sort(const bitvectors &data, const bitvectors &queries, const Matrixf &oriQueries, const Matrixf &centroids, const Matrixi &clusterMembers, const int k, int atleastCluster) {
  const int centroids_size = centroids.size();
  
  std::vector<std::vector<IdxDistPair>> answers(queries.size());
  for (int idx_q = 0; idx_q < (int)queries.size(); idx_q++) {
    std::vector<IdxDistPairFloat> nearestCluster(centroids_size);
    
    for (int c_idx=0; c_idx<centroids_size; c_idx++) {
      nearestCluster[c_idx] = IdxDistPairFloat(c_idx, euclideanDistNoSQRT(oriQueries[idx_q], centroids[c_idx]));
    }
    std::sort(nearestCluster.begin(), nearestCluster.end(),
      [](const IdxDistPairFloat &a, const IdxDistPairFloat &b) -> bool {
        return a.dist < b.dist;
      }
    );

    int nearestClusterCumulative = 0;
    int farthestClusterIdx = 0;
    for (const IdxDistPairFloat& nc: nearestCluster) {
      farthestClusterIdx++;
      nearestClusterCumulative += clusterMembers[nc.idx].size();
      if (nearestClusterCumulative >= k) {
        break;
      }
    }
    while (farthestClusterIdx < atleastCluster) {
      nearestClusterCumulative += clusterMembers[nearestCluster[farthestClusterIdx].idx].size();
      farthestClusterIdx++;
    }

    std::vector<int> pairsIdx(nearestClusterCumulative);
    std::vector<uint32_t> pairsDist(nearestClusterCumulative);
    int p_idx = 0;
    for (int i=0; i<farthestClusterIdx; i++) {
      for (int cm: clusterMembers[nearestCluster[i].idx]) {
        pairsIdx[p_idx] = cm;
        pairsDist[p_idx] = hammingDist(queries[idx_q], data[cm]);
        p_idx++;
      }
    }

    answers[idx_q] = BitVecEngine::KNNFromDistsIndicesSupplied(pairsDist.data(), pairsIdx.data(), nearestClusterCumulative, k);
  }

  return answers;
}

std::vector<std::vector<IdxDistPair>> queryWithClusterInfo_heap_early_abandon(const bitvectors &data, const bitvectors &queries, const Matrixf &oriQueries, const Matrixf &centroids, const Matrixi &clusterMembers, const int k, int atleastCluster) {
  const int centroids_size = centroids.size();
  
  auto comparator = [](IdxDistPair const& a, IdxDistPair const& b) -> bool {
    return a.dist < b.dist;
  };
  
  std::vector<std::vector<IdxDistPair>> answers(queries.size());
  for (int idx_q = 0; idx_q < (int)queries.size(); idx_q++) {
    std::vector<IdxDistPairFloat> nearestCluster(centroids_size);
    
    for (int c_idx=0; c_idx<centroids_size; c_idx++) {
      nearestCluster[c_idx] = IdxDistPairFloat(c_idx, euclideanDistNoSQRT(oriQueries[idx_q], centroids[c_idx]));
    }
    std::sort(nearestCluster.begin(), nearestCluster.end(),
      [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
        return a.dist < b.dist;
      }
    );

    int nearestClusterCumulative = 0;
    int farthestClusterIdx = 0;
    for (const IdxDistPairFloat& nc: nearestCluster) {
      farthestClusterIdx++;
      nearestClusterCumulative += clusterMembers[nc.idx].size();
      if (nearestClusterCumulative >= k) {
        break;
      }
    }

    while (farthestClusterIdx < atleastCluster) {
      nearestClusterCumulative += clusterMembers[nearestCluster[farthestClusterIdx].idx].size();
      farthestClusterIdx++;
    }

    std::vector<IdxDistPair> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    uint32_t bsfK = 0;
    uint32_t dist;

    int iter = 0;
    for (int i=0; i<farthestClusterIdx; i++) {
      for (int cm: clusterMembers[nearestCluster[i].idx]) {
        if (iter < k) {
          dist = hammingDist(queries[idx_q], data[cm]);
          pairs.push_back(IdxDistPair(cm, dist));
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          if (dist > bsfK) {
            bsfK = dist;
          }
        } else{
          dist = hammingDistEarlyAbandon(queries[idx_q], data[cm], bsfK);
          if (dist < bsfK) {
            pairs.push_back(IdxDistPair(cm, dist));
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            std::pop_heap(pairs.begin(), pairs.end(), comparator);
            pairs.pop_back();
            bsfK = (pairs.front()).dist;
          }
        }
        iter++;
      }
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }

  return answers;
}

std::vector<std::vector<IdxDistPair>> queryWithClusterInfo_sort_early_abandon(const bitvectors &data, const bitvectors &queries, const Matrixf &oriQueries, const Matrixf &centroids, const Matrixi &clusterMembers, const int k, int atleastCluster) {
  const int centroids_size = centroids.size();
  
  std::vector<std::vector<IdxDistPair>> answers(queries.size());
  for (int idx_q = 0; idx_q < (int)queries.size(); idx_q++) {
    std::vector<IdxDistPairFloat> nearestCluster(centroids_size);
    
    for (int c_idx=0; c_idx<centroids_size; c_idx++) {
      nearestCluster[c_idx] = IdxDistPairFloat(c_idx, euclideanDistNoSQRT(oriQueries[idx_q], centroids[c_idx]));
    }
    std::sort(nearestCluster.begin(), nearestCluster.end(),
      [](const IdxDistPairFloat &a, const IdxDistPairFloat &b) -> bool {
        return a.dist < b.dist;
      }
    );

    int nearestClusterCumulative = 0;
    int farthestClusterIdx = 0;
    for (const IdxDistPairFloat& nc: nearestCluster) {
      farthestClusterIdx++;
      nearestClusterCumulative += clusterMembers[nc.idx].size();
      if (nearestClusterCumulative >= k) {
        break;
      }
    }
    while (farthestClusterIdx < atleastCluster) {
      nearestClusterCumulative += clusterMembers[nearestCluster[farthestClusterIdx].idx].size();
      farthestClusterIdx++;
    }

    std::vector<IdxDistPair> pairs;
    pairs.reserve(k+1);
    auto insertionSort = [&pairs](int idxStart, int idx, uint32_t dist) {
      if (pairs.size() > 0) {
        int i = idxStart;
        for (; i>=0; i--) {
          if (dist > pairs[i].dist) {
            break;
          }
        }
        pairs.emplace(pairs.begin() + i + 1, idx, dist);
      } else {
        pairs.emplace(pairs.begin(), idx, dist);
      }
    };

    int p_idx = 0;
    uint32_t dist, bsfK = 0;
    for (int i=0; i<farthestClusterIdx; i++) {
      for (int cm: clusterMembers[nearestCluster[i].idx]) {
        if (p_idx < k) {
          dist = hammingDist(queries[idx_q], data[cm]);
          if (dist > bsfK) {
            bsfK = dist;
          }
          insertionSort(p_idx, cm, dist);
        } else {
          dist = hammingDistEarlyAbandon(queries[idx_q], data[cm], bsfK);
          if (dist < bsfK) {
            insertionSort(k-1, cm, dist);
            pairs.pop_back();
            bsfK = pairs[k-1].dist;
          }
        }
        p_idx++;
      }
    }

    pairs.shrink_to_fit();
    answers[idx_q] = std::move(pairs);
  }

  return answers;
}

/**
 * Rerank function
 */
void rerankUsingEuclideanDistance(const int queryNRows, const Matrixf &oriQueries, const Matrixf &originalData, const int kTemp, const int k, const std::vector<std::vector<IdxDistPair>> &tempResult, std::vector<std::vector<IdxDistPairFloat>> &answers) {
  for (int idx_q = 0; idx_q < queryNRows; idx_q++) {
    std::vector<int> pairsIdx(kTemp);
    std::vector<float> pairsDist(kTemp);
    
    for (int i=0; i < kTemp; i++) {
      pairsIdx[i] = tempResult[idx_q][i].idx;
      pairsDist[i] = euclideanDistNoSQRT(oriQueries[idx_q], originalData[tempResult[idx_q][i].idx]);
    }

    answers[idx_q] = BitVecEngine::KNNFromDistsIndicesSupplied(pairsDist.data(), pairsIdx.data(), kTemp, k);
  }
}

void rerankUsingEAEuclideanDistance(const int queryNRows, const Matrixf &oriQueries, const Matrixf &originalData, const int k, const std::vector<std::vector<IdxDistPair>> &tempResult, std::vector<std::vector<IdxDistPairFloat>> &answers) {
  auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
    return a.dist < b.dist;
  };

  for (int idx_q = 0; idx_q < queryNRows; idx_q++) {
    float bsfK = 0;
    float dist;
    
    std::vector<IdxDistPairFloat> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    int i=0;
    for (const auto &tr: tempResult[idx_q]) {
      if (i < k) {
        dist = euclideanDistNoSQRT(oriQueries[idx_q], originalData[tr.idx]);
        pairs.push_back(IdxDistPairFloat(tr.idx, dist));
        std::push_heap(pairs.begin(), pairs.end(), comparator);
        if (dist > bsfK) {
          bsfK = dist;
        }
      } else {
        dist = euclideanDistEarlyAbandon(oriQueries[idx_q], originalData[tr.idx], bsfK);
        if (dist < bsfK) {
          pairs.push_back(IdxDistPairFloat(tr.idx, dist));
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          std::pop_heap(pairs.begin(), pairs.end(), comparator);
          pairs.pop_back();
          bsfK = (pairs.front()).dist;
        }
      }
      i += 1;
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }
}

/**
 * Helper Function for Query END
 */

std::vector<std::vector<IdxDistPair>> BitVecEngine::query(const bitvectors &queries, const int k, int method) const {
  if (method == QueryMethod::Sort) {
    return query_sort(this->data, queries, k);
  } else if (method == QueryMethod::Heap) {
    return query_heap(this->data, queries, k);
  } else if (method == QueryMethod::HeapEarlyAbandon) {
    return query_heap_early_abandon(this->data, queries, k);
  } else {  // method == QueryMethod::SortEarlyAbandon
    return query_sort_early_abandon(this->data, queries, k);
  }
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryRerank(const bitvectors &queries, const Matrixf &oriQueries, const int k, const int factor, int method, bool earlyAbandonRank) const {
  const int kTemp = (factor * k) > (int)data.size() ? data.size() : factor * k;
  std::vector<std::vector<IdxDistPair>> tempResult = this->query(queries, kTemp, method);

  // rerank using euclidean distance
  const int queryNRows = oriQueries.size();
  std::vector<std::vector<IdxDistPairFloat>> answers(queryNRows);
  if (earlyAbandonRank) {
    rerankUsingEAEuclideanDistance(queryNRows, oriQueries, originalData, k, tempResult, answers);
  } else {
    rerankUsingEuclideanDistance(queryNRows, oriQueries, originalData, kTemp, k, tempResult, answers);
  }

  return answers;
}

std::vector<std::vector<IdxDistPair>> BitVecEngine::queryWithClusterInfo(const bitvectors &queries, const Matrixf &oriQueries, const int k, int method, int atleastCluster) const {
  if (method == QueryMethod::Sort) {
    return queryWithClusterInfo_sort(
      this->data, queries, oriQueries, this->centroids, this->clusterMembers, k,  atleastCluster);
  } else if (method == QueryMethod::Heap) {
    return queryWithClusterInfo_heap(
      this->data, queries, oriQueries, this->centroids, this->clusterMembers, k, atleastCluster);
  } else if (method == QueryMethod::HeapEarlyAbandon) {
    return queryWithClusterInfo_heap_early_abandon(
      this->data, queries, oriQueries, this->centroids, this->clusterMembers, k, atleastCluster);
  } else {  // method == QueryMethod::SortEarlyAbandon
    return queryWithClusterInfo_sort_early_abandon(
      this->data, queries, oriQueries, this->centroids, this->clusterMembers, k, atleastCluster);
  }
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryRerankWithClusterInfo(const bitvectors &queries, const Matrixf &oriQueries, const int k, const int factor, int method, int atleastCluster, bool earlyAbandonRank) const {
  const int kTemp = (factor * k) > (int)data.size() ? data.size() : factor * k;
  std::vector<std::vector<IdxDistPair>> tempResult = 
    queryWithClusterInfo(queries, oriQueries, kTemp, method, atleastCluster);

  // rerank using euclidean distance
  const int queryNRows = oriQueries.size();
  std::vector<std::vector<IdxDistPairFloat>> answers(queryNRows);
  if (earlyAbandonRank) {
    rerankUsingEAEuclideanDistance(queryNRows, oriQueries, originalData, k, tempResult, answers);
  } else {
    rerankUsingEuclideanDistance(queryNRows, oriQueries, originalData, kTemp, k, tempResult, answers);
  }

  return answers;
}


/**
 * Naive
 */
std::vector<std::vector<IdxDistPairFloat>> queryNaive_sort(const Matrixf &data, const Matrixf &queries, const int k) {
  const int dataNRows = data.size();
  std::vector<std::vector<IdxDistPairFloat>> answers(queries.size());

  int q_idx = 0;
  for (const std::vector<float>& q: queries) {
    std::vector<float> pairsDist(dataNRows);
    
    for (int dataIndex = 0; dataIndex < dataNRows; dataIndex++) {
      pairsDist[dataIndex] = euclideanDistNoSQRT(q, data[dataIndex]);
    }

    answers[q_idx] = BitVecEngine::KNNFromDists(pairsDist.data(), dataNRows, k);
    q_idx++;
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> queryNaive_sort_early_abandon(const Matrixf &data, const Matrixf &queries, const int k) {
  const int dataNRows = data.size();
  std::vector<std::vector<IdxDistPairFloat>> answers(queries.size());
  
  int q_idx = 0;
  for (const std::vector<float>& q: queries) {
    std::vector<IdxDistPairFloat> pairs;
    pairs.reserve(k+1);

    auto insertionSort = [&pairs](int idxStart, int idx, float dist) {
      if (pairs.size() > 0) {
        int i = idxStart-1;
        for (; i>0; i--) {
          if (dist > pairs[i-1].dist) {
            break;
          }
        }
        pairs.emplace(pairs.begin() + i, idx, dist);
      } else {
        pairs.emplace(pairs.begin(), idx, dist);
      }
    };
    
    float dist, bsfK = 0;
    int dataIndex = 0;
    for (; dataIndex < k; dataIndex++) {
      dist = euclideanDistNoSQRT(q, data[dataIndex]);
      if (dist > bsfK) {
        bsfK = dist;
      }
      insertionSort(dataIndex, dataIndex, dist);
    }
    
    for (; dataIndex < dataNRows; dataIndex++) {
      dist = euclideanDistEarlyAbandon(q, data[dataIndex], bsfK);
      if (dist < bsfK) {
        insertionSort(k, dataIndex, dist);
        pairs.pop_back();
        bsfK = pairs[k-1].dist;
      }
    }

    pairs.shrink_to_fit();
    answers[q_idx] = std::move(pairs);
    q_idx++;
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> queryNaive_heap(const Matrixf &data, const Matrixf &queries, const int k) {
  std::vector<std::vector<IdxDistPairFloat>> answers(queries.size());
  auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
    return a.dist < b.dist;
  };

  for (int idx_q=0; idx_q<(int)queries.size(); idx_q++) {
    std::vector<IdxDistPairFloat> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    
    int iter = 0;
    for (const std::vector<float>& datum: data) {
      pairs.push_back(IdxDistPairFloat(iter, euclideanDistNoSQRT(queries[idx_q], datum)));
      std::push_heap(pairs.begin(), pairs.end(), comparator);
      if (iter+1 > k) {
        std::pop_heap(pairs.begin(), pairs.end(), comparator);
        pairs.pop_back();
      }
      iter++;
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> queryNaive_heap_early_abandon(const Matrixf &data, const Matrixf &queries, const int k) {
  const int queryNRows = queries.size();
  std::vector<std::vector<IdxDistPairFloat>> answers(queryNRows);
  auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
    return a.dist < b.dist;
  };

  for (int idx_q=0; idx_q<queryNRows; idx_q++) {
    std::vector<IdxDistPairFloat> pairs;
    std::make_heap(pairs.begin(), pairs.end(), comparator);
    float dist, bsfK = 0;
    
    int iter = 0;
    for (const std::vector<float>& datum: data) {
      if (iter < k) {
        dist = euclideanDistNoSQRT(queries[idx_q], datum);
        pairs.push_back(IdxDistPairFloat(iter, dist));
        std::push_heap(pairs.begin(), pairs.end(), comparator);
        if (dist > bsfK)  {
          bsfK = dist;
        }
      } else {
        dist = euclideanDistEarlyAbandon(queries[idx_q], datum, bsfK);
        if (dist < bsfK) {
          pairs.push_back(IdxDistPairFloat(iter, dist));
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          std::pop_heap(pairs.begin(), pairs.end(), comparator);
          pairs.pop_back();
          bsfK = (pairs.front()).dist;
        }
      }
      iter++;
    }
    std::sort_heap(pairs.begin(), pairs.end(), comparator);
    answers[idx_q] = std::move(pairs);
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryNaive(const Matrixf &oriQueries, const int k, int method) const {
  if (method == QueryMethod::Sort) {
    return queryNaive_sort(originalData, oriQueries, k);
  } else if (method == QueryMethod::Heap) {
    return queryNaive_heap(originalData, oriQueries, k);
  } else if (method == QueryMethod::HeapEarlyAbandon) {
    return queryNaive_heap_early_abandon(originalData, oriQueries, k);
  } else {  // method == QueryMethod::SortEarlyAbandon
    return queryNaive_sort_early_abandon(originalData, oriQueries, k);
  }
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryNaiveTriangleInequality(const Matrixf &oriQueries, const int k, int method) const {
  std::vector<std::vector<IdxDistPairFloat>> answers(oriQueries.size());
  auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
    return a.dist < b.dist;
  };
  std::vector<float> qToCCDist(this->centroids_size);
  long totalPruned = 0;
  if (method == QueryMethod::Heap) {
    for (int q_idx=0; q_idx < (int)oriQueries.size(); q_idx++) {
      for (int i=0; i<centroids_size; i++) {
        qToCCDist[i] = euclideanDist(oriQueries[q_idx], this->centroids[i]);
      }

      Eigen::VectorXi qToCCIdx = Eigen::VectorXi::LinSpaced(centroids_size, 0, centroids_size-1);
      std::sort(qToCCIdx.data(), qToCCIdx.data()+qToCCIdx.size(),
        [&qToCCDist](int i, int j) -> bool {
          return qToCCDist[i] < qToCCDist[j];
        }
      );

      std::vector<IdxDistPairFloat> pairs;
      pairs.reserve(k+1);
      std::make_heap(pairs.begin(), pairs.end(), comparator);
      float bsfK = 0;

      long prunedPerQuery = 0;
      int counter = 0;
      for (int ccIdxIdx = 0; ccIdxIdx < qToCCIdx.size(); ccIdxIdx++) {
        const int clusterIdx = qToCCIdx[ccIdxIdx];
        const int clusterDataStartIdx = this->clusterMembersStartIdx[clusterIdx];
        if (this->clusterMembers[clusterIdx].size() == 0) {
          continue;
        }
        const float qToCCDistance = qToCCDist[clusterIdx];

        int interCounter = 0;
        int interIndex = clusterDataStartIdx;
        for (const int dataIndex: clusterMembers[clusterIdx]) {
          if (counter < k) {
            float dist = euclideanDist(oriQueries[q_idx], this->originalData[interIndex]);
            pairs.emplace_back(dataIndex, dist);
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            if (dist > bsfK) {
              bsfK = dist;
            }
            counter++;
          } else {
            if (bsfK <= qToCCDistance - datapointToCCDist[dataIndex]) {
              prunedPerQuery += clusterMembers[clusterIdx].size() - interCounter;
              break;
            }
            float dist = euclideanDist(oriQueries[q_idx], originalData[interIndex]);
            if (dist < bsfK) {
              pairs.emplace_back(dataIndex, dist);
              std::push_heap(pairs.begin(), pairs.end(), comparator);
              std::pop_heap(pairs.begin(), pairs.end(), comparator);
              pairs.pop_back();
              bsfK = (pairs.front()).dist;
            }
          }
          interIndex++;
          interCounter++;
        }
      }
      
      std::sort_heap(pairs.begin(), pairs.end(), comparator);
      answers[q_idx] = std::move(pairs);

      totalPruned += prunedPerQuery;
    }
  } else if (method == QueryMethod::HeapEarlyAbandon) {

  } else if (method == QueryMethod::Sort) {
    for (int q_idx=0; q_idx < (int)oriQueries.size(); q_idx++) {
      for (int i=0; i<centroids_size; i++) {
        qToCCDist[i] = euclideanDist(oriQueries[q_idx], this->centroids[i]);
      }

      Eigen::VectorXi qToCCIdx = Eigen::VectorXi::LinSpaced(centroids_size, 0, centroids_size-1);
      std::sort(qToCCIdx.data(), qToCCIdx.data()+qToCCIdx.size(),
        [&qToCCDist](int i, int j) -> bool {
          return qToCCDist[i] < qToCCDist[j];
        }
      );

      std::vector<IdxDistPairFloat> pairs;
      pairs.reserve(k+1);

      auto insertionSort = [&pairs](int idxStart, int idx, float dist) {
        if (pairs.size() > 0) {
          int i = idxStart-1;
          for (; i>0; i--) {
            if (dist > pairs[i-1].dist) {
              break;
            }
          }
          pairs.emplace(pairs.begin() + i, idx, dist);
        } else {
          pairs.emplace(pairs.begin(), idx, dist);
        }
      };
      
      float bsfK = 0;
      long prunedPerQuery = 0;
      int counter = 0;
      for (int ccIdxIdx = 0; ccIdxIdx < qToCCIdx.size(); ccIdxIdx++) {
        const int clusterIdx = qToCCIdx[ccIdxIdx];
        const int clusterDataStartIdx = this->clusterMembersStartIdx[clusterIdx];
        if (this->clusterMembers[clusterIdx].size() == 0) {
          continue;
        }
        const float qToCCDistance = qToCCDist[clusterIdx];

        int interCounter = 0;
        int interIndex = clusterDataStartIdx;
        for (const int dataIndex: clusterMembers[clusterIdx]) {
          if (counter < k) {
            float dist = euclideanDist(oriQueries[q_idx], originalData[interIndex]);
            if (dist > bsfK) {
              bsfK = dist;
            }
            insertionSort(counter, dataIndex, dist);
            counter++;
          } else {
            if (bsfK <= qToCCDistance - datapointToCCDist[dataIndex]) {
              prunedPerQuery += clusterMembers[clusterIdx].size() - interCounter;
              break;
            }

            float dist = std::sqrt(euclideanDistEarlyAbandon(oriQueries[q_idx], originalData[interIndex], bsfK*bsfK));
            if (dist < bsfK) {
              insertionSort(k, dataIndex, dist);
              pairs.pop_back();
              bsfK = pairs[k-1].dist;
            }
          }
          interIndex++;
          interCounter++;
        }
      }
      
      pairs.shrink_to_fit();
      answers[q_idx] = std::move(pairs);

      totalPruned += prunedPerQuery;
    }
  }
  std::cout << "total pruned: " << totalPruned << std::endl;
  std::cout << "average pruned per query: " << totalPruned / oriQueries.size() << std::endl;

  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryNaiveWithClusterInfo(const Matrixf &oriQueries, const int k, int method, int atleastCluster) const {
  const int centroids_size = centroids.size();

  std::vector<std::vector<IdxDistPairFloat>> answers(oriQueries.size());
  for (int idx_q = 0; idx_q < (int)oriQueries.size(); idx_q++) {
    std::vector<IdxDistPairFloat> nearestCluster(centroids_size);
    
    for (int c_idx=0; c_idx<centroids_size; c_idx++) {
      nearestCluster[c_idx] = IdxDistPairFloat(c_idx, euclideanDistNoSQRT(oriQueries[idx_q], centroids[c_idx]));
    }
    std::sort(nearestCluster.begin(), nearestCluster.end(),
      [](const IdxDistPairFloat &a, const IdxDistPairFloat &b) -> bool {
        return a.dist < b.dist;
      }
    );

    int nearestClusterCumulative = 0;
    int farthestClusterIdx = 0;
    for (const IdxDistPairFloat& nc: nearestCluster) {
      farthestClusterIdx++;
      nearestClusterCumulative += clusterMembers[nc.idx].size();
      if (nearestClusterCumulative >= k) {
        break;
      }
    }

    while (farthestClusterIdx < atleastCluster) {
      nearestClusterCumulative += clusterMembers[nearestCluster[farthestClusterIdx].idx].size();
      farthestClusterIdx++;
    }
    
    if (method == QueryMethod::Sort) {
      std::vector<int> pairsIdx(nearestClusterCumulative);
      std::vector<float> pairsDist(nearestClusterCumulative);
      int p_idx = 0;
      for (int i=0; i<farthestClusterIdx; i++) {
        for (int cm: clusterMembers[nearestCluster[i].idx]) {
          pairsIdx[p_idx] = cm;
          pairsDist[p_idx] = euclideanDistNoSQRT(oriQueries[idx_q], originalData[cm]);
          p_idx++;
        }
      }

      answers[idx_q] = BitVecEngine::KNNFromDistsIndicesSupplied(pairsDist.data(), pairsIdx.data(), nearestClusterCumulative, k);
    } else if (method == QueryMethod::Heap) {
      auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
        return a.dist < b.dist;
      };

      std::vector<IdxDistPairFloat> pairs;
      std::make_heap(pairs.begin(), pairs.end(), comparator);
      int iter = 0;
      for (int i=0; i<farthestClusterIdx; i++) {
        for (int cm: clusterMembers[nearestCluster[i].idx]) {
          pairs.push_back(IdxDistPairFloat(cm, euclideanDistNoSQRT(oriQueries[idx_q], originalData[cm])));
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          if (iter+1 > k) {
            std::pop_heap(pairs.begin(), pairs.end(), comparator);
            pairs.pop_back();
          }
          iter++;
        }
      }
      std::sort_heap(pairs.begin(), pairs.end(), comparator);
      answers[idx_q] = std::move(pairs);

    } else if (method == QueryMethod::HeapEarlyAbandon) {
      auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
        return a.dist < b.dist;
      };
      
      std::vector<IdxDistPairFloat> pairs;
      std::make_heap(pairs.begin(), pairs.end(), comparator);
      float dist, bsfK = 0;

      int iter = 0;
      for (int i=0; i<farthestClusterIdx; i++) {
        for (int cm: clusterMembers[nearestCluster[i].idx]) {
          if (iter < k) {
            dist = euclideanDistNoSQRT(oriQueries[idx_q], originalData[cm]);
            pairs.push_back(IdxDistPairFloat(cm, dist));
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            if (dist > bsfK) {
              bsfK = dist;
            }
          } else{
            dist = euclideanDistEarlyAbandon(oriQueries[idx_q], originalData[cm], bsfK);
            if (dist < bsfK) {
              pairs.push_back(IdxDistPairFloat(cm, dist));
              std::push_heap(pairs.begin(), pairs.end(), comparator);
              std::pop_heap(pairs.begin(), pairs.end(), comparator);
              pairs.pop_back();
              bsfK = (pairs.front()).dist;
            }
          }
          iter++;
        }
      }
      std::sort_heap(pairs.begin(), pairs.end(), comparator);
      answers[idx_q] = std::move(pairs);
    } else {  // method == QueryMethod::SortEarlyAbandon
      std::vector<IdxDistPairFloat> pairs;
      pairs.reserve(k+1);
      
      auto insertionSort = [&pairs](int idxStart, int idx, float dist) {
        if (pairs.size() > 0) {
          int i = idxStart;
          for (; i>=0; i--) {
            if (dist > pairs[i].dist) {
              break;
            }
          }
          pairs.emplace(pairs.begin() + i + 1, idx, dist);
        } else {
          pairs.emplace(pairs.begin(), idx, dist);
        }
      };

      int p_idx = 0;
      float dist, bsfK = 0;

      for (int i=0; i<farthestClusterIdx; i++) {
        for (int cm: clusterMembers[nearestCluster[i].idx]) {
          if (p_idx < k) {
            dist = euclideanDistNoSQRT(oriQueries[idx_q], originalData[cm]);
            if (dist > bsfK) {
              bsfK = dist;
            }
            insertionSort(p_idx, cm, dist);
          } else {
            dist = euclideanDistEarlyAbandon(oriQueries[idx_q], originalData[cm], bsfK);
            if (dist < bsfK) {
              insertionSort(k-1, cm, dist);
              pairs.pop_back();
              bsfK = pairs[k-1].dist;
            }
          }
          p_idx++;
        }
      }

      pairs.shrink_to_fit();
      answers[idx_q] = std::move(pairs);
    }
  }

  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryNaiveWithClusterInfoParallelDiskResident(const std::string datasetFilepath, const Matrixf &oriQueries, const std::vector<int> &clustIdx, const int k, int thread, int method, int batch, int atleastCluster) const {
  std::vector<std::vector<IdxDistPairFloat>> globalAnswers;
  FILE *infile = fopen(datasetFilepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return std::vector<std::vector<IdxDistPairFloat>>();
  }

  int loadedData_offset = 0;
  int dimen = oriQueries[0].size();
  while (true) {
    Matrixf loadedData;
    int loadedData_iter = 0;
    while (loadedData_iter < batch) {
      std::vector<float> v(dimen);
      if (fread(v.data(), sizeof(float), dimen, infile) == 0) {
        break;
      }
      loadedData.push_back(v);
      loadedData_iter++;
    }

    if (loadedData_iter == 0) {
      break;
    }

    Matrixi batchClusterMembers(centroids.size());
    for (int ci=loadedData_offset; ci < loadedData_offset + loadedData_iter; ci++) {
      batchClusterMembers[clustIdx[ci]].push_back(ci);
    }

    std::vector<std::vector<IdxDistPairFloat>> answers;

    const int queryLen = oriQueries.size();
    if (method == QueryMethod::Sort) {
      #pragma omp parallel num_threads(thread)
      {
        std::vector<std::vector<IdxDistPairFloat>> privateAnswer;

        #pragma omp for nowait schedule(static)
        for (int qIter = 0; qIter < queryLen; qIter++) {
          // centroids info
          const int centroids_size = centroids.size();
          std::vector<IdxDistPairFloat> nearestCluster(centroids_size);

          for (int c_idx=0; c_idx<centroids_size; c_idx++) {
            nearestCluster[c_idx] = IdxDistPairFloat(c_idx, euclideanDistNoSQRT(oriQueries[qIter], centroids[c_idx]));
          }
          std::sort(nearestCluster.begin(), nearestCluster.end(),
            [](const IdxDistPairFloat &a, const IdxDistPairFloat &b) -> bool {
              return a.dist < b.dist;
            }
          );

          int nearestClusterCumulative = 0;
          int farthestClusterIdx = 0;
          for (const IdxDistPairFloat& nc: nearestCluster) {
            farthestClusterIdx++;
            nearestClusterCumulative += batchClusterMembers[nc.idx].size();
            if (nearestClusterCumulative >= k) {
              break;
            }
          }
          while (farthestClusterIdx < atleastCluster) {
            nearestClusterCumulative += clusterMembers[nearestCluster[farthestClusterIdx].idx].size();
            farthestClusterIdx++;
          }

          std::vector<int> pairsIdx(nearestClusterCumulative);
          std::vector<float> pairsDist(nearestClusterCumulative);
          int p_idx = 0;
          for (int i=0; i<farthestClusterIdx; i++) {
            for (int cm: batchClusterMembers[nearestCluster[i].idx]) {
              pairsIdx[p_idx] = cm;
              pairsDist[p_idx] = euclideanDistNoSQRT(oriQueries[qIter], loadedData[cm - loadedData_offset]);
              p_idx++;
            }
          }

          privateAnswer.push_back(BitVecEngine::KNNFromDistsIndicesSupplied(pairsDist.data(), pairsIdx.data(), nearestClusterCumulative, k));
        }

        const int num_threads = omp_get_num_threads();
        #pragma omp for schedule(static) ordered
        for (int i=0; i<num_threads; i++) {
          #pragma omp ordered
          answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
        }
      }
    }
    if (loadedData_offset == 0) {
      globalAnswers.insert(globalAnswers.end(), answers.begin(), answers.end());
    } else {
      for (int qIter=0; qIter < queryLen; qIter++) {
        globalAnswers[qIter].insert(globalAnswers[qIter].end(), answers[qIter].begin(), answers[qIter].end());
        std::sort(globalAnswers[qIter].begin(), globalAnswers[qIter].end(), 
          [](const IdxDistPairFloat &a, const IdxDistPairFloat &b) -> bool {
            return a.dist < b.dist;
          }
        );
        globalAnswers[qIter].resize(k);
      }
    }

    if (loadedData_iter < batch) {
      break;
    }
    loadedData_offset += loadedData_iter;
  }
  
  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
  
  return globalAnswers;
}


/**
 * Query with progressive filtering
 */
std::vector<std::vector<IdxDistPair>> BitVecEngine::queryFiltering_Sort(const bitvectors &queries, const int k) const {
  std::vector<std::vector<IdxDistPair>> answers(queries.size());

  int q_idx = 0;
  for (const bitv& q: queries) {
    
    // split Q
    std::vector<bitv> QSplitted(SubVector);
    for (int i=0; i<SubVector; i++) {
      for (int j=0; j<SubVectorLen; j++) {
        QSplitted[i].resize(N/SubVector);
        QSplitted[i][j] = q[j + i * SubVectorLen];
      }
    }

    auto hamming = [&QSplitted, this](const int subvectorIdx, const int dataIdx) -> uint32_t{
      uint32_t setBits = 0;
      for (int iter=0; iter<SubVectorLen; iter++) {
        setBits += __builtin_popcountl(QSplitted[subvectorIdx][iter] ^ dataSplitted[subvectorIdx][dataIdx][iter]);
      }

      return setBits;
    };
    
    const int dataLen = data.size();
    int filterCounter = 0;
    int hammingTemp;
    std::vector<IdxDistPair> pairs(dataLen);
    for (int dataIndex=0; dataIndex<dataLen; dataIndex++) {  
      hammingTemp = hamming(0, dataIndex);
      filterCounter += (hammingTemp == 0) ? 1 : 0;
      pairs[dataIndex] = IdxDistPair(dataIndex, hammingTemp);
    }

    // filter data by each subvector added distance
    for (int i=0; i<SubVector; i++) {
      if (i > 0) {
        filterCounter = 0;
        for (IdxDistPair& p: pairs) {
          p.dist += hamming(i, p.idx);
          filterCounter += (p.dist == 0) ? 1 : 0;
        }
      }

      std::sort(pairs.begin(), pairs.end(), 
        [](const IdxDistPair &a, const IdxDistPair &b) -> bool {
          return a.dist < b.dist;
        }
      );

      std::cout << "filtercounter: " << filterCounter << std::endl;
      if (filterCounter <= k || i == SubVector-1) {
        pairs.resize(k);
        break;
      }
      pairs.resize(filterCounter);
    }
    
    answers[q_idx] = std::move(pairs);
    q_idx++;
  }
  
  return answers;
}

std::vector<std::vector<IdxSubDistPair>> BitVecEngine::queryFiltering_Heap(const bitvectors &queries, const int k) const {
  std::vector<std::vector<IdxSubDistPair>> answers(queries.size());

  int q_idx = 0;
  for (const bitv& q: queries) {
    auto hamming = [&q, this](const int dataIdx, const int subvectorIdx) -> int {
      const int min = subvectorIdx * SubVectorLen;
      const int max = min + SubVectorLen;
      int setBits = 0;
      for (int i=min; i<max; i++) {
        setBits += __builtin_popcount(q[i] ^ data[dataIdx][i]);
      }

      return setBits;
    };

    auto progressiveComparator = [&q, this, &hamming](IdxSubDistPair &a, IdxSubDistPair &b) -> bool {
      int subvecIdx = 0;
      while (subvecIdx+1 < SubVector && a.dist[subvecIdx] == b.dist[subvecIdx]) {
        subvecIdx++;
        if (a.dist[subvecIdx] == -1) {
          a.dist[subvecIdx] = hamming(a.idx, subvecIdx);
        }
        if (b.dist[subvecIdx] == -1) {
          b.dist[subvecIdx] = hamming(b.idx, subvecIdx);
        }
      };
      return a.dist[subvecIdx] < b.dist[subvecIdx];
    };
    std::vector<IdxSubDistPair> pairs;
    std::make_heap(pairs.begin(), pairs.end(), progressiveComparator);
    
    const int datasetSize = data.size();
    for (int iter=0; iter<datasetSize; iter++) {
      pairs.push_back(IdxSubDistPair(
        SubVector, iter, hamming(iter, 0)
      ));
      std::push_heap(pairs.begin(), pairs.end(), progressiveComparator);
      if (iter+1 > k) {
        std::pop_heap(pairs.begin(), pairs.end(), progressiveComparator);
        pairs.pop_back();
      }
    }
    std::sort_heap(pairs.begin(), pairs.end(), progressiveComparator);
    answers[q_idx] = std::move(pairs);
    q_idx++;
  }
  
  return answers;
}

/**
 * Query with parallel wise
 */
std::vector<std::vector<IdxDistPair>> BitVecEngine::queryParallel(const bitvectors &queries, const int k, const int thread) const {
  std::vector<std::vector<IdxDistPair>> answers;

  const int queryLen = queries.size();
  #pragma omp parallel num_threads(thread)
  {
    std::vector<std::vector<IdxDistPair>> privateAnswer;
    auto comparator = [](IdxDistPair const& a, IdxDistPair const& b) -> bool {
      return a.dist < b.dist;
    };

    #pragma omp for nowait schedule(static)
    for (int qIter = 0; qIter < queryLen; qIter++) {
      std::vector<IdxDistPair> pairs;
      bitv q = queries[qIter];
      std::make_heap(pairs.begin(), pairs.end(), comparator);
      
      int iter = 0;
      for (const bitv& datum: data) {
        pairs.push_back(IdxDistPair(iter, hammingDist(q, datum)));
        std::push_heap(pairs.begin(), pairs.end(), comparator);
        if (iter+1 > k) {
          std::pop_heap(pairs.begin(), pairs.end(), comparator);
          pairs.pop_back();
        }
        iter++;
      }
      std::sort_heap(pairs.begin(), pairs.end(), comparator);
      privateAnswer.push_back(pairs);
    }

    const int num_threads = omp_get_num_threads();
    #pragma omp for schedule(static) ordered
    for (int i=0; i<num_threads; i++) {
      #pragma omp ordered
      answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
    }
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryNaiveParallel(const Matrixf &queries, const int k, const int thread, int method) const {
  std::vector<std::vector<IdxDistPairFloat>> answers;

  const int queryLen = queries.size();
  if (method == QueryMethod::HeapEarlyAbandon) {
    #pragma omp parallel num_threads(thread)
    {
      std::vector<std::vector<IdxDistPairFloat>> privateAnswer;
      auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
        return a.dist < b.dist;
      };

      #pragma omp for nowait schedule(static)
      for (int qIter = 0; qIter < queryLen; qIter++) {
        std::vector<IdxDistPairFloat> pairs;
        std::make_heap(pairs.begin(), pairs.end(), comparator);
        float bsfK = 0, dist;
        
        int iter = 0;
        for (const std::vector<float>& datum: originalData) {
          if (iter < k) {
            dist = euclideanDistNoSQRT(queries[qIter], datum);
            pairs.push_back(IdxDistPairFloat(iter, dist));
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            if (dist > bsfK) {
              bsfK = dist;
            }
          } else {
            dist = euclideanDistEarlyAbandon(queries[qIter], datum, bsfK);
            if (dist < bsfK) {
              pairs.push_back(IdxDistPairFloat(iter, dist));
              std::push_heap(pairs.begin(), pairs.end(), comparator);
              std::pop_heap(pairs.begin(), pairs.end(), comparator);
              pairs.pop_back();
              bsfK = (pairs.front()).dist;
            }
          }
          iter++;
        }
        std::sort_heap(pairs.begin(), pairs.end(), comparator);
        privateAnswer.push_back(pairs);
      }

      const int num_threads = omp_get_num_threads();
      #pragma omp for schedule(static) ordered
      for (int i=0; i<num_threads; i++) {
        #pragma omp ordered
        answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
      }
    }
  } else if (method == QueryMethod::SortEarlyAbandon) {
    #pragma omp parallel num_threads(thread)
    {
      std::vector<std::vector<IdxDistPairFloat>> privateAnswer;

      #pragma omp for nowait schedule(static)
      for (int qIter = 0; qIter < queryLen; qIter++) {
        std::vector<IdxDistPairFloat> pairs;
        pairs.reserve(k+1);

        auto insertionSort = [&pairs](int idxStart, int idx, float dist) {
          if (pairs.size() > 0) {
            int i = idxStart;
            for (; i>=0; i--) {
              if (dist > pairs[i].dist) {
                break;
              }
            }
            pairs.emplace(pairs.begin() + i + 1, idx, dist);
          } else {
            pairs.emplace(pairs.begin(), idx, dist);
          }
        };
        float bsfK = 0, dist;
        
        int iter = 0;
        for (const std::vector<float>& datum: originalData) {
          if (iter < k) {
            dist = euclideanDistNoSQRT(queries[qIter], datum);
            if (dist > bsfK) {
              bsfK = dist;
            }
            insertionSort(iter, iter, dist);
          } else {
            dist = euclideanDistEarlyAbandon(queries[qIter], datum, bsfK);
            if (dist < bsfK) {
              insertionSort(k-1, iter, dist);
              pairs.pop_back();
              bsfK = pairs[k-1].dist;
            }
          }
          iter++;
        }

        pairs.shrink_to_fit();
        privateAnswer.push_back(pairs);
      }

      const int num_threads = omp_get_num_threads();
      #pragma omp for schedule(static) ordered
      for (int i=0; i<num_threads; i++) {
        #pragma omp ordered
        answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
      }
    }
  } else if (method == QueryMethod::Sort) {
    #pragma omp parallel num_threads(thread)
    {
      std::vector<std::vector<IdxDistPairFloat>> privateAnswer;

      #pragma omp for nowait schedule(static)
      for (int qIter = 0; qIter < queryLen; qIter++) {
        std::vector<float> pairsDist(originalData.size());
        
        int iter = 0;
        for (const std::vector<float>& datum: originalData) {
          pairsDist[iter] = euclideanDistNoSQRT(datum, queries[qIter]);
          iter++;
        }

        privateAnswer.push_back(BitVecEngine::KNNFromDists(pairsDist.data(), pairsDist.size(), k));
      }

      const int num_threads = omp_get_num_threads();
      #pragma omp for schedule(static) ordered
      for (int i=0; i<num_threads; i++) {
        #pragma omp ordered
        answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
      }
    }
  }
  
  return answers;
}

std::vector<std::vector<IdxDistPairFloat>> BitVecEngine::queryNaiveParallelDiskResident(const std::string datasetFilepath, const Matrixf &queries, const int k, const int thread, int method, int batch) const {
  std::vector<std::vector<IdxDistPairFloat>> globalAnswers;
  FILE *infile = fopen(datasetFilepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return std::vector<std::vector<IdxDistPairFloat>>();
  }

  int loadedData_offset = 0;
  int dimen = queries[0].size();
  while (true) {
    Matrixf loadedData;
    int loadedData_iter = 0;
    while (loadedData_iter < batch) {
      std::vector<float> v(dimen);
      if (fread(v.data(), sizeof(float), dimen, infile) == 0) {
        break;
      }
      loadedData.push_back(v);
      loadedData_iter++;
    }

    if (loadedData_iter == 0) {
      break;
    }

    std::vector<std::vector<IdxDistPairFloat>> answers;

    const int queryLen = queries.size();
    if (method == QueryMethod::HeapEarlyAbandon) {
      #pragma omp parallel num_threads(thread)
      {
        std::vector<std::vector<IdxDistPairFloat>> privateAnswer;
        auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
          return a.dist < b.dist;
        };

        #pragma omp for nowait schedule(static)
        for (int qIter = 0; qIter < queryLen; qIter++) {
          std::vector<IdxDistPairFloat> pairs;
          std::make_heap(pairs.begin(), pairs.end(), comparator);
          float bsfK = 0, dist;
          
          int iter = 0;
          for (const std::vector<float>& datum: loadedData) {
            if (iter < k) {
              dist = euclideanDistNoSQRT(queries[qIter], datum);
              pairs.push_back(IdxDistPairFloat(iter+loadedData_offset, dist));
              std::push_heap(pairs.begin(), pairs.end(), comparator);
              if (dist > bsfK) {
                bsfK = dist;
              }
            } else {
              dist = euclideanDistEarlyAbandon(queries[qIter], datum, bsfK);
              if (dist < bsfK) {
                pairs.push_back(IdxDistPairFloat(iter+loadedData_offset, dist));
                std::push_heap(pairs.begin(), pairs.end(), comparator);
                std::pop_heap(pairs.begin(), pairs.end(), comparator);
                pairs.pop_back();
                bsfK = (pairs.front()).dist;
              }
            }
            iter++;
          }
          std::sort_heap(pairs.begin(), pairs.end(), comparator);
          privateAnswer.push_back(pairs);
        }

        const int num_threads = omp_get_num_threads();
        #pragma omp for schedule(static) ordered
        for (int i=0; i<num_threads; i++) {
          #pragma omp ordered
          answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
        }
      }
    } else if (method == QueryMethod::SortEarlyAbandon) {
      #pragma omp parallel num_threads(thread)
      {
        std::vector<std::vector<IdxDistPairFloat>> privateAnswer;

        #pragma omp for nowait schedule(static)
        for (int qIter = 0; qIter < queryLen; qIter++) {
          std::vector<IdxDistPairFloat> pairs;
          pairs.reserve(k+1);

          auto insertionSort = [&pairs](int idxStart, int idx, float dist) {
            if (pairs.size() > 0) {
              int i = idxStart;
              for (; i>=0; i--) {
                if (dist > pairs[i].dist) {
                  break;
                }
              }
              pairs.emplace(pairs.begin() + i + 1, idx, dist);
            } else {
              pairs.emplace(pairs.begin(), idx, dist);
            }
          };
          float bsfK = 0, dist;
          
          int iter = 0;
          for (const std::vector<float>& datum: loadedData) {
            if (iter < k) {
              dist = euclideanDistNoSQRT(queries[qIter], datum);
              if (dist > bsfK) {
                bsfK = dist;
              }
              insertionSort(iter, iter+loadedData_offset, dist);
            } else {
              dist = euclideanDistEarlyAbandon(queries[qIter], datum, bsfK);
              if (dist < bsfK) {
                insertionSort(k-1, iter+loadedData_offset, dist);
                pairs.pop_back();
                bsfK = pairs[k-1].dist;
              }
            }
            iter++;
          }

          pairs.shrink_to_fit();
          privateAnswer.push_back(pairs);
        }

        const int num_threads = omp_get_num_threads();
        #pragma omp for schedule(static) ordered
        for (int i=0; i<num_threads; i++) {
          #pragma omp ordered
          answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
        }
      }
    } else if (method == QueryMethod::Sort) {
      #pragma omp parallel num_threads(thread)
      {
        std::vector<std::vector<IdxDistPairFloat>> privateAnswer;

        #pragma omp for nowait schedule(static)
        for (int qIter = 0; qIter < queryLen; qIter++) {
          std::vector<float> pairsDist(loadedData.size());
          std::vector<int> pairsIdx(loadedData.size());
          
          int iter = 0;
          for (const std::vector<float>& datum: loadedData) {
            pairsIdx[iter] = iter+loadedData_offset;
            pairsDist[iter] = euclideanDistNoSQRT(datum, queries[qIter]);
            iter++;
          }

          privateAnswer.push_back(BitVecEngine::KNNFromDistsIndicesSupplied(pairsDist.data(), pairsIdx.data(), loadedData.size(), k));
        }

        const int num_threads = omp_get_num_threads();
        #pragma omp for schedule(static) ordered
        for (int i=0; i<num_threads; i++) {
          #pragma omp ordered
          answers.insert(answers.end(), privateAnswer.begin(), privateAnswer.end());
        }
      }
    }
    if (loadedData_offset == 0) {
      globalAnswers.insert(globalAnswers.end(), answers.begin(), answers.end());
    } else {
      for (int qIter=0; qIter < (int)globalAnswers.size(); qIter++) {
        globalAnswers[qIter].insert(globalAnswers[qIter].begin(), answers[qIter].begin(), answers[qIter].end());
        std::sort(globalAnswers[qIter].begin(), globalAnswers[qIter].end(), 
          [](const IdxDistPairFloat &a, const IdxDistPairFloat &b) -> bool {
            return a.dist < b.dist;
          }
        );
        globalAnswers[qIter].resize(k);
      }
    }

    if (loadedData_iter < batch) {
      break;
    }
    loadedData_offset += loadedData_iter;
  }
  
  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
  
  return globalAnswers;
}

bitv BitVecEngine::getBitV(const int idx) {
  return data[idx];
}

void BitVecEngine::appendBitV(const bitvectors &bitVectors) {
  data.insert(data.end(), bitVectors.begin(), bitVectors.end());
}

void BitVecEngine::deleteBitV(int idx) {
  data.erase(data.begin() + idx);
}
