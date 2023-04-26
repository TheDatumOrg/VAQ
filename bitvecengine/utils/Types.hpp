#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <vector>
#include <cstdint>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "Heap.hpp"

/* Eigen Types Util */
template<class T, int Rows=Eigen::Dynamic, int Cols=Eigen::Dynamic>
using RowMatrix = Eigen::Matrix<T, Rows, Cols, Eigen::RowMajor>;
using RowMatrixXf = RowMatrix<float>;

template<class T, int Rows=Eigen::Dynamic, int Cols=Eigen::Dynamic>
using ColMatrix = Eigen::Matrix<T, Rows, Cols, Eigen::ColMajor>;

template<class T>
using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;

template<class T>
using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RowVectorXf = RowVector<float>;

using Matrixf = std::vector<std::vector<float>>;
using Matrixi = std::vector<std::vector<int>>;

using CodebookType = RowMatrix<uint16_t>;
template<class T=uint8_t>
using CodebookTypeColMajor = ColMatrix<T>;
using CentroidsMatType = RowMatrix<float>;
using CentroidsPerSubsType = std::vector<CentroidsMatType, Eigen::aligned_allocator<CentroidsMatType > >;
using CentroidsPerSubsTypeColMajor = std::vector<ColMatrix<float>, Eigen::aligned_allocator<ColMatrix<float> > >;
using LUTType = ColMatrix<float>;
using SmallLUTType = ColMatrix<uint8_t>;

/* Index-Distance Pair Type */
template<typename DistType>
struct IdxDistPairBase {
  int idx;
  DistType dist;

  IdxDistPairBase() = default;
  IdxDistPairBase(const IdxDistPairBase<DistType> &rhs) = default;
  IdxDistPairBase(int _idx, DistType _dist) : idx(_idx), dist(_dist) {}
};

using IdxDistPair = IdxDistPairBase<uint32_t>;
using IdxDistPairUint8 = IdxDistPairBase<uint8_t>;
using IdxDistPairUint16 = IdxDistPairBase<uint16_t>;
using IdxDistPairInt16 = IdxDistPairBase<int16_t>;
using IdxDistPairFloat = IdxDistPairBase<float>;
using IdxDistPairDouble = IdxDistPairBase<double>;
using IdxDistPairComplexFloat = IdxDistPairBase<std::complex<float>>;

template<typename A, typename B>
std::vector<IdxDistPairBase<B>> vectorIdxDistPairConverter(std::vector<IdxDistPairBase<A>> a) {
  std::vector<IdxDistPairBase<B>> b(a.size());
  for (int i=0; i<(int)a.size(); i++) {
    b[i] = { a[i].idx, static_cast<B>(a[i].dist) };
  }
  return b;
}
template<typename T>
void vectorIdxDistPairConverterFloatHeap(std::vector<IdxDistPairBase<T>> a, f::float_maxheap_t &res, int res_offset) {
  int * heap_ids = res.ids + res_offset;
  float * heap_dis = res.val + res_offset;
  for (int i=0; i<(int)a.size(); i++) {
    heap_ids[i] = a[i].idx;
    heap_dis[i] = static_cast<float>(a[i].dist);
  }
}

struct IdxSubDistPair {
  std::vector<int> dist;
  int idx;

  IdxSubDistPair() {};
  IdxSubDistPair(int _subvector, int _idx, int _firstDist) : dist(_subvector, -1), idx(_idx){
    dist[0] = _firstDist;
  }
};

template<typename T>
static inline void printIdxDistPair(std::vector<IdxDistPairBase<T>> &pairs) {
  std::cout << "Index Dist Pair content" << std::endl;
  const int len = pairs.size();
  for (int i=0; i<len; i++) {
    std::cout << "\t(" << pairs[i].idx << ", " << pairs[i].dist << ")" << std::endl;
  }
  std::cout << std::endl;
}

/* Label-Distance Pair vector */
template<typename T = float>
struct LabelDistVec {
  std::vector<int> labels;
  std::vector<T> distances;
};

using LabelDistVecF = LabelDistVec<>;


#endif