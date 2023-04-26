#ifndef VAQ_H_
#define VAQ_H_

#include <iostream>
#include <sstream>

#include <Eigen/Eigenvalues>
#include "glpk.h"

#include "KMeans.hpp"

#include "utils/Types.hpp"
#ifdef __AVX2__
#include "utils/AVXUtils.hpp"
#endif
#include "utils/Math.hpp"
#include "utils/Experiment.hpp"
#include "utils/Heap.hpp"

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

extern "C" {
#ifndef FINTEGER
#define FINTEGER int
#endif

int sgemm_ (
        const char *transa, const char *transb, 
        FINTEGER *m, FINTEGER *n, FINTEGER *k, 
        const float *alpha, const float *a,
        FINTEGER *lda, const float *b,
        FINTEGER *ldb, float *beta,
        float *c, FINTEGER *ldc);
}

class VAQ {
public:
  struct NNMethod { // mostly prune method
    enum { 
      Sort   = 0x01u, //0b00000001
      EA     = 0x02u, //0b00000010 Early Abandon
      TI     = 0x04u, //0b00000100 Triangle Inequality Prune
      Fast   = 0x08u, //0b00001000 shuffle SIMD
      Fast2  = 0x10u, //0b00010000 gather SIMD
      Fast3  = 0x20u, //0b00100000 shuffle and gather SIMD
      Fast4  = 0x40u, //0b01000000 sort and shuffle SIMD
      Heap   = 0x80u  //0b10000000 heap
    };
  };

  int mBitBudget;
  int mSubspaceNum;
  float mPercentVarExplained;
  int mMinBitsPerSubs, mMaxBitsPerSubs;
  uint32_t mMethods;
  
  RowMatrix<Eigen::scomplex> mEigenVectors;
  #ifndef VAQ_OPTIMIZE
  RowMatrix<float> mEigenVectorsReal;
  #endif
  // Maybe for next project
  #if 0
  RowMatrix<Eigen::scomplex> mEigenVectorsBeforeBalancing;
  #endif
  Eigen::VectorXf mCumSumVarExplainedPerSubs;
  CentroidsPerSubsType mCentroidsPerSubs;
  CentroidsPerSubsTypeColMajor mCentroidsPerSubsCMajor;
  std::vector<int> mBitsAlloc, mCentroidsNum;
  int mSubsLen;
  int mHighestSubs, mTotalDim;
  int mXTrainRows, mXTrainCols;
  CodebookType mCodebook;
  CodebookTypeColMajor<> mCodebookCMajor;
  CodebookTypeColMajor<uint16_t> mCodebookCMajor16;

  // TI Prune Vars
  int mTIClusterNum, mTISegmentNum = -1;
  float mTIVariance = 1;
  RowMatrixXf mTIClusters;
  std::vector<std::vector<int>> mTIClustersMember;
  std::vector<int> mClusterMembersStartIdx;
  std::vector<float> mCodeToCCDist;
  RowMatrixXf metadata;
  float mVisit = 1;
  bool mHierarchicalKmeans = false;
  bool mBinaryKmeans = false;

  // SIMD Search Vars
  RowVector<float> mOffsets;
  ColVector<float> mScale;
  int mStartShufIdx = 0;

  void train(RowMatrixXf &XTrain, bool verbose = false);

  void encode(const RowMatrixXf &XTrain);
  
  template<class T>
  void encodeImpl(const RowMatrixXf &XTrain, T &codebook);

  void encodeImplFast3(const RowMatrixXf &XTrain);

  LabelDistVecF search(const RowMatrixXf &XTest, const int k, bool verbose=false);

  LabelDistVecF refine(const RowMatrixXf &XTest, const LabelDistVecF &answersIn, const RowMatrixXf &XTrain, const int k);

  void clusterTI(bool useKMeans = false, bool verbose = false);
  
  void learnQuantization(const RowMatrixXf &XTrain, float sampleRatio = 0.1f);

  void parseMethodString(std::string methodString);

  inline uint32_t searchMethod() const {
    return mMethods;
  }

private:
  /**
   * train Auxiliary Functions
   */
  std::vector<std::vector<int>> getBelongsToCluster(const RowMatrixXf &X, const CentroidsMatType &C);
  std::vector<int> getBelongsToBinaryCluster(const RowMatrixXf &X, const CentroidsMatType &C, size_t &sizeleft, size_t &sizeright);
  CentroidsMatType hierarchicalBinKmeans(RowMatrixXf& X, int depth, const int maxdepth);

  /**
   * search Auxiliary Functions
   */
#ifdef __AVX2__
  template<int maxbit=8>
  void CreateLUT(const RowVectorXf &query, LUTType &lut) {
    lut.setZero();
    
    static constexpr int packet_width = 8; // objs per simd register
    for (int subs=0; subs<mHighestSubs; subs++) {
      if (mCentroidsNum[subs] >= 8) { // only vectorized when centroids >= 8
        const int nstripes = (int)(std::ceil(mCentroidsNum[subs] / packet_width));
        __m256 accumulators[(1 << maxbit)/8];  // max centroids
        auto lut_ptr = lut.data() + lut.rows()*subs;

        for (int i=0; i<nstripes; i++) {
          accumulators[i] = _mm256_setzero_ps();
        }

        for (int j=0; j<mSubsLen; j++) {
          auto centroids_ptr = (mCentroidsPerSubsCMajor[subs]).data() + mCentroidsPerSubsCMajor[subs].rows()*j;

          auto q_broadcast = _mm256_set1_ps(query(subs * mSubsLen + j));
          for (int i=0; i<nstripes; i++) {
            auto centroids_col = _mm256_load_ps(centroids_ptr);
            centroids_ptr += packet_width;

            auto diff = _mm256_sub_ps(q_broadcast, centroids_col);
            accumulators[i] = fma(diff, diff, accumulators[i]);
          }
        }

        // write out dists in this col of the lut
        for (uint16_t i=0; i<nstripes; i++) {
          _mm256_store_ps((float *)lut_ptr, accumulators[i]);
          lut_ptr += packet_width;
        }
      } else {
        float * lutPtr = lut.data() + lut.rows() * subs;
        const float* qsub = query.data() + subs * mSubsLen;
        fvec_L2sqr_ny(lutPtr, qsub, mCentroidsPerSubs[subs].data(), mSubsLen, mCentroidsNum[subs]);
      }
    }
  }
#else
  template<int maxbit=8>
  void CreateLUT(const RowVectorXf &query, LUTType &lut) {
    lut.setZero();
    
    float * lutPtr = lut.data();
    for (int subs=0; subs<mHighestSubs; subs++) {
      const float* qsub = query.data() + subs * mSubsLen;
      fvec_L2sqr_ny(lutPtr, qsub, mCentroidsPerSubs[subs].data(), mSubsLen, mCentroidsNum[subs]);
      lutPtr += lut.rows();
    }
  }
#endif
  
  void CreateLUTFast3(const RowVectorXf &query, LUTType &lut, const int nShuffleNum, const int nGatherDim);
  
  void searchTriangleInequality(LUTType &lut, const int k, uint32_t methods, const Eigen::VectorXi &qToCCIdx, const std::vector<float> &qToCCDist, long &prunedPerQuery, int q_idx, f::float_maxheap_t &res);
  
  void searchEarlyAbandon(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res);
  
  void searchHeap(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res);

  void searchSort(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res);
  
  std::vector<IdxDistPairInt16> searchFast(LUTType &lut, const int k);

  void searchFast2(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res);

  void searchFast3(LUTType &lutGather, const int k, int q_idx, f::float_maxheap_t &res);

  inline RowMatrixXf ProjectOnEigenVectors(const RowMatrixXf &X, bool withChecking=false) {
    if (!withChecking) {
      return (X * mEigenVectors).real();
    }
    RowMatrix<Eigen::scomplex> XComplex = X * mEigenVectors;
    for (int i=0; i<XComplex.rows(); i++) {
      for (int j=0; j<XComplex.cols(); j++) {
        if (XComplex(i, j).imag() != 0 || std::isnan(XComplex(i, j).real()) || std::isinf(XComplex(i, j).real())) {
          XComplex(i, j).real(0);
        }
      }
    }
    
    return XComplex.real();
  }

  inline RowMatrixXf ProjectOnFirstDimsEigenVectors(const RowMatrixXf &X, int firstdims, bool withChecking=false) {
    if (!withChecking) {
      return (X * mEigenVectors.block(0, 0, mEigenVectors.rows(), firstdims)).real();
    }
    RowMatrix<Eigen::scomplex> XComplex = X * mEigenVectors.block(0, 0, mEigenVectors.rows(), firstdims);
    for (int i=0; i<XComplex.rows(); i++) {
      for (int j=0; j<XComplex.cols(); j++) {
        if (XComplex(i, j).imag() != 0 || std::isnan(XComplex(i, j).real()) || std::isinf(XComplex(i, j).real())) {
          XComplex(i, j).real(0);
        }
      }
    }
    
    return XComplex.real();
  }
  
  inline void ProjectOnEigenVectorsInPlace(RowMatrixXf &X, bool withChecking=false) {
    const int bs = 256 * 1024;
    if (X.rows() > bs) {
      int batchNum = X.rows() / bs;
      if (X.rows() % bs > 0) {
        batchNum += 1;
      }
      // #pragma omp parallel for num_threads(1)
      for (int b=0; b<batchNum; b++) {
        int currBatchRows = bs;
        if ((b == batchNum-1) && (X.rows() % bs > 0)) {
          currBatchRows = X.rows() % bs;
        }
        #ifdef VAQ_OPTIMIZE
        RowMatrix<Eigen::scomplex> blockProjected = (X.block(b*bs, 0, currBatchRows, X.cols()) * mEigenVectors);
        if (withChecking) {
          for (int i=0; i<blockProjected.rows(); i++) {
            for (int j=0; j<blockProjected.cols(); j++) {
              if (blockProjected(i, j).imag() != 0 || std::isnan(blockProjected(i, j).real()) || std::isinf(blockProjected(i, j).real())) {
                blockProjected(i, j).real(0);
              }
            }
          }
        }
        X.block(b*bs, 0, currBatchRows, X.cols()) = blockProjected.real();
        #else

        // X.block(b*bs, 0,   currBatchRows, X.cols()) = X.block(b*bs, 0, currBatchRows, X.cols()) * mEigenVectorsReal;
        RowMatrix<float> blockProjected(currBatchRows, X.cols());
        FINTEGER Arow = currBatchRows, Acol = X.cols(), 
                 Brow = X.cols(), Bcol = X.cols();
        float one = 1, zero = 0;
        sgemm_(
          "Not transposed", "Not transposed",
          &Bcol, &Arow, &Brow,
          &one, mEigenVectorsReal.data(), &Bcol,
          X.row(b * bs).data(), &Acol,
          &zero, blockProjected.data(), &Bcol);
        X.block(b*bs, 0, currBatchRows, X.cols()) = blockProjected;
        #endif
      }
    } else {
      X = (X * mEigenVectors).real();
    }
  }

  inline void ProjectOnFirstDimsEigenVectorsInPlace(RowMatrixXf &X, int firstdims, bool withChecking=false) {
    const int batchSize = 10000000;
    if (X.rows() > batchSize) {
      int batchNum = X.rows() / batchSize;
      if (X.rows() % batchSize > 0) {
        batchNum += 1;
      }
      for (int b=0; b<batchNum; b++) {
        int currBatchRows = batchSize;
        if ((b == batchNum-1) && (X.rows() % batchSize > 0)) {
          currBatchRows = X.rows() % batchSize;
        }
        RowMatrix<Eigen::scomplex> blockProjected = (X.block(b*batchSize, 0, currBatchRows, X.cols()) * mEigenVectors.block(0, 0, mEigenVectors.rows(), firstdims));
        if (withChecking) {
          for (int i=0; i<blockProjected.rows(); i++) {
            for (int j=0; j<blockProjected.cols(); j++) {
              if (blockProjected(i, j).imag() != 0 || std::isnan(blockProjected(i, j).real()) || std::isinf(blockProjected(i, j).real())) {
                blockProjected(i, j).real(0);
              }
            }
          }
        }
        X.block(b*batchSize, 0, currBatchRows, X.cols()) = blockProjected.real();
      }
    } else {
      std::cout << "ashiap " << X.rows() << " " << X.cols() << std::endl;
      std::cout << "santuy " << mEigenVectors.rows() << " " << mEigenVectors.cols() << std::endl;
      X = (X * mEigenVectors.block(0, 0, mEigenVectors.rows(), firstdims)).real();
    }
  }  
};

#endif  // BITVECENGINE_H_