#ifndef BITVECENGINE_H_
#define BITVECENGINE_H_

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <initializer_list>
#include <assert.h>
#include <algorithm>
#include <complex>

#include <unistd.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>

#include <iostream>
#include <stdio.h>


#include "KMeans.hpp"
#include "glpk.h"
#include "BitVector.hpp"
#include "utils/DistanceFunctions.hpp"
#include "utils/Types.hpp"
#include "utils/Experiment.hpp"
#include "utils/AVXUtils.hpp"
#include "utils/TimingUtils.hpp"

class BitVecEngine {
private:
  // LUT information
  Eigen::MatrixXcf eigenVectors;
  CentroidsMatType centroidsMat;
  std::vector<int> solutionX;
  std::vector<int> LUTBitPosition;
  int nonZeroAllocCount;

  // clusters VAQ information
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> clustersVAQ;
  std::vector<float> codeToCCDist;
  Matrixi clusterMembersVAQ;
  std::vector<int> subspacesBeginIdx;


  // group VAQ information
  std::vector<int> group_bits;
  Matrixi groupMembersVAQ;
  std::vector<int> groupMembersStartIdx;

  inline Eigen::MatrixXf ProjectOnEigenVectors(const Eigen::MatrixXf &Z, bool withChecking=false) const {
    Eigen::MatrixXcf ZxV2 = Z * this->eigenVectors;
    Eigen::MatrixXf ZRet(Z.rows(), Z.cols());
    if (!withChecking) {
      ZRet = ZxV2.real();
    } else {
      for (int i=0; i<ZxV2.rows(); i++) {
        for (int j=0; j<ZxV2.cols(); j++) {
          if (ZxV2(i, j).imag() != 0 || std::isnan(ZxV2(i, j).real()) || std::isinf(ZxV2(i, j).real())) {
            ZRet(i, j) = 0;
          } else {
            ZRet(i, j) = ZxV2(i, j).real();
          }
        }
      }
    }

    return ZRet;
  }

public:
  struct NNMethod { // mostly prune method
    enum { 
      Sort  = 0x01u, //0b00000001
      EA    = 0x02u, //0b00000010 Early Abandon
      TI    = 0x04u, //0b00000100 Triangle Inequality Prune
    };
  };

  struct QueryMethod {
    enum { Heap, Sort, HeapEarlyAbandon, SortEarlyAbandon };
  };

  const int N;  // bitvlen
  const int actBitVLen;
  const int SubVector;  // number of subvector, (split factor)
  const int SubVectorLen;
  bitvectors data;
  Matrixf originalData;
  std::vector<bitvectors> dataSplitted;
  std::vector<int> hardcodedAllocPerSubspace;
  std::vector<int> hardcodedSolutionX;
  
  // cluster information
  bitvectors centroidsBin;
  Matrixf centroids;
  int centroids_size;
  Matrixi clusterMembers;
  // Triangle Inequality prune
  std::vector<float> datapointToCCDist;
  std::vector<int> clusterMembersStartIdx;

  BitVecEngine(int _N, int _SubVector=1) 
    : N(_N), actBitVLen((N + 63) / 64), SubVector(_SubVector), SubVectorLen(actBitVLen/SubVector), dataSplitted(SubVector) {}
  
  // Auxiliary methods
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

  void parseAndLoadHardcoded(std::string allocpersubspace, std::string solutionx) {
    auto parseUtil = [](std::string inpstr) -> std::vector<int> {
      std::vector<int> ret;
      ret.reserve(64);

      std::vector<std::string> parsed;
      std::stringstream ss(inpstr);
      while (ss.good()) {
        std::string substr;
        std::getline(ss, substr, ',');
        parsed.push_back(substr);
      }

      for (std::string token: parsed) {
        ret.push_back(std::stoi(token));
      }

      return ret;
    };

    hardcodedAllocPerSubspace = parseUtil(allocpersubspace);
    hardcodedSolutionX = parseUtil(solutionx);
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
  static inline std::vector<IdxDistPairBase<T>> KNNFromDistsIndicesSupplied(const T* dists, const int *indices, int len, int k) {
    std::vector<IdxDistPairBase<T>> ret(k);
    for (int i = 0; i < k; i++) {
      ret[i] = IdxDistPairBase<T>{indices[i], dists[i]};
    }
    
    std::sort(ret.begin(), ret.end(), 
      [](const IdxDistPairBase<T>& a, const IdxDistPairBase<T>& b) -> bool {
        return a.dist < b.dist;
      }
    );

    for (int i = k; i < len; i++) {
      maybeInsertNeighbor(ret, IdxDistPairBase<T>{indices[i], dists[i]});
    }
    return ret;
  }


  static void binaryEncodingSimple(const Eigen::MatrixXf &XTrain, const Eigen::MatrixXf &XTest, bitvectors &XTrainBinary, bitvectors &XTestBinary) {
    /* Compute PCA on training data */
    Eigen::EigenSolver<Eigen::MatrixXf> es(XTrain.transpose() * XTrain);
    
    // get sorted index
    std::vector<IdxDistPairComplexFloat> eigvalueidx(es.eigenvalues().rows());
    for (int i = 0; i < es.eigenvalues().rows(); i++) {
      eigvalueidx[i].dist = es.eigenvalues()(i);
      eigvalueidx[i].idx = i;
    }

    std::sort(eigvalueidx.begin(), eigvalueidx.end(),
      [](const IdxDistPairComplexFloat &a, const IdxDistPairComplexFloat &b) -> bool {
        if (a.dist.real() == b.dist.real()) {
          return a.dist.imag() > b.dist.imag();
        }
        return a.dist.real() > b.dist.real();
      }
    );

    Eigen::MatrixXcf V2(es.eigenvectors().rows(), es.eigenvectors().cols());
    for (int i = 0; i < V2.cols(); i++) {
      V2.col(i) = es.eigenvectors().col(eigvalueidx[i].idx);
    }

    auto ProjectOnEigenVectors = [&V2](const Eigen::MatrixXf &Z, bool withChecking=false) -> Eigen::MatrixXf {
      Eigen::MatrixXcf ZxV2 = Z * V2;
      Eigen::MatrixXf ZRet(ZxV2.rows(), ZxV2.cols());
      if (!withChecking) {
        ZRet = ZxV2.real();
      } else {
        for (int i=0; i<ZxV2.rows(); i++) {
          for (int j=0; j<ZxV2.cols(); j++) {
            if (ZxV2(i, j).imag() != 0 || std::isnan(ZxV2(i, j).real()) || std::isinf(ZxV2(i, j).real())) {
              ZRet(i, j) = 0;
            } else {
              ZRet(i, j) = ZxV2(i, j).real();
            }
          }
        }
      }

      return ZRet;
    };

    /* New representation: Project data to learned eigenvectors V2 */
    Eigen::MatrixXf XTrainChecked = ProjectOnEigenVectors(XTrain);
    Eigen::MatrixXf XTestChecked = ProjectOnEigenVectors(XTest);

    for (int i=0; i<XTrainChecked.rows(); i++) {
      uint64_t temp = 0;
      int j;
      for (j=1; j<=XTrainChecked.cols(); j++) {
        temp |= (uint64_t) ((XTrainChecked(i, j-1) > 0) ? 1 : 0);
        if (j % 64 == 0) {
          XTrainBinary[i][j / 64 - 1] = temp;
          temp = 0;
        } else {
          temp <<= 1;
        }
      }

      if (XTrainChecked.cols() % 64 != 0) {
        XTrainBinary[i][j / 64] = temp << (64 - (j % 64));
      }
    }

    for (int i=0; i<XTestChecked.rows(); i++) {
      uint64_t temp = 0;
      int j;
      for (j=1; j<=XTestChecked.cols(); j++) {
        temp |= (uint64_t) ((XTestChecked(i, j-1) > 0) ? 1 : 0);
        if (j % 64 == 0) {
          XTestBinary[i][j / 64 - 1] = temp;
          temp = 0;
        } else {
          temp <<= 1;
        }
      }

      if (XTestChecked.cols() % 64 != 0) {
        XTestBinary[i][j / 64] = temp << (64 - (j % 64));
      }
    }
  }
  
  static void binaryEncoding(const Eigen::MatrixXf &XTrain, const Eigen::MatrixXf &XTest, bitvectors &XTrainBinary, bitvectors &XTestBinary, int bitBudget) {
    /* Compute PCA on training data */
    Eigen::EigenSolver<Eigen::MatrixXf> es(XTrain.transpose() * XTrain);
    
    // get sorted index
    std::vector<IdxDistPairComplexFloat> eigvalueidx(es.eigenvalues().rows());
    for (int i = 0; i < es.eigenvalues().rows(); i++) {
      eigvalueidx[i].dist = es.eigenvalues()(i);
      eigvalueidx[i].idx = i;
    }

    std::sort(eigvalueidx.begin(), eigvalueidx.end(),
      [](const IdxDistPairComplexFloat &a, const IdxDistPairComplexFloat &b) -> bool {
        if (a.dist.real() == b.dist.real()) {
          return a.dist.imag() > b.dist.imag();
        }
        return a.dist.real() > b.dist.real();
      }
    );

    Eigen::MatrixXcf V2(es.eigenvectors().rows(), es.eigenvectors().cols());
    for (int i = 0; i < V2.cols(); i++) {
      V2.col(i) = es.eigenvectors().col(eigvalueidx[i].idx);
    }

    auto ProjectOnEigenVectors = [&V2](const Eigen::MatrixXf &Z, bool withChecking=false) -> Eigen::MatrixXf {
      Eigen::MatrixXcf ZxV2 = Z * V2;
      Eigen::MatrixXf ZRet(ZxV2.rows(), ZxV2.cols());
      if (!withChecking) {
        ZRet = ZxV2.real();
      } else {
        for (int i=0; i<ZxV2.rows(); i++) {
          for (int j=0; j<ZxV2.cols(); j++) {
            if (ZxV2(i, j).imag() != 0 || std::isnan(ZxV2(i, j).real()) || std::isinf(ZxV2(i, j).real())) {
              ZRet(i, j) = 0;
            } else {
              ZRet(i, j) = ZxV2(i, j).real();
            }
          }
        }
      }

      return ZRet;
    };

    /* New representation: Project data to learned eigenvectors V2 */
    Eigen::MatrixXf XTrainChecked = ProjectOnEigenVectors(XTrain);
    Eigen::MatrixXf XTestChecked = ProjectOnEigenVectors(XTest);
    
    /* Assign bits to dimensions */
    Eigen::VectorXf varExplainedSum(eigvalueidx.size());
    for (int i=0; i<(int)eigvalueidx.size(); i++) {
      varExplainedSum(i) = eigvalueidx[i].dist.real();
    }
    varExplainedSum /= varExplainedSum.sum();

    for (int i=0; i<varExplainedSum.rows(); i++) {
      if (varExplainedSum(i) < 1e-11) {
        varExplainedSum(i) = 1e-17;
      }
    }

    // MILP
    glp_prob *lp = glp_create_prob();
    glp_smcp parm; glp_init_smcp(&parm);
    parm.meth = GLP_DUAL;
    // parm.pricing = GLP_PT_STD;
    // parm.r_test = GLP_RT_STD;
    // parm.presolve = GLP_ON;
    glp_set_obj_dir(lp, GLP_MAX);
    int glp_rows = varExplainedSum.rows(), glp_cols = varExplainedSum.rows();
    glp_add_rows(lp, glp_rows);
    glp_add_cols(lp, glp_cols);
    
    // set integer constraint
    for (int d=1; d<=glp_cols; d++) {
      glp_set_col_kind(lp, d, GLP_IV);
    }

    // std::cout << "varExplainedSum: " << std::endl;
    for (int i=0; i<glp_cols; i++) {
      // std::cout << varExplainedSum(i) << ";";
      glp_set_obj_coef(lp, i+1, varExplainedSum(i));
    }
    // std::cout << std::endl;

    auto cumsum = [](const Eigen::VectorXf &Z) -> std::vector<float> {
      std::vector<float> cumSumVar(Z.rows());
      cumSumVar[0] = Z(0);
      for (int i=1; i<Z.rows(); i++) {
        cumSumVar[i] = Z(i) + cumSumVar[i-1];
      }

      return cumSumVar;
    };

    std::vector<float> cumSumVar = cumsum(varExplainedSum);
    float UniformAllocVarExplained = (bitBudget <= (int)cumSumVar.size()) ? cumSumVar[bitBudget-1] : cumSumVar.back();
    float varExplainedXPercentage = 0.99 * UniformAllocVarExplained;

    int lastMatIdx = 1, rowCounter = 1;
    std::vector<int> rowIndices(glp_cols * glp_cols + 1, 0), colIndices(glp_cols * glp_cols + 1, 0);
    std::vector<double> numVal(glp_cols * glp_cols + 1, 0);

    // Bit allocation per dimension 0 or more
    // x_i >= 0 or -x_i <=0
    // x_i <= 8
    for (int i=1; i<=glp_cols; i++) {
      double lb = 0;
      if (i <= bitBudget) {
        lb = (varExplainedXPercentage >= cumSumVar[i-1]) ? 1.0 : 0.0;
      }
      glp_set_col_bnds(lp, i, GLP_DB, lb, 8.0);
    }

    // sum(x_i) = budget
    // 1 1 1 1 1 = budget
    glp_set_row_bnds(lp, rowCounter, GLP_FX, bitBudget, 0.0);
    for (int d = 1; d <= glp_cols; d++) {
      rowIndices[lastMatIdx] = rowCounter;
      colIndices[lastMatIdx] = d;
      numVal[lastMatIdx] = 1.0;
      lastMatIdx++;
    }
    rowCounter++;

    auto nextpow2 = [](double x) -> int {
      if (x == 0) {
        return 0;
      }
      
      return (int)std::pow(2, std::floor(std::log2(std::abs(x))));
    };
    
    // Allocate bits based on the explained variance
    // 2^(x_i) >= ratioVar 2^(x_i+1) 
    // k = nextpow2(ratioVar)
    // or
    // x_i - x_(i+1) <= k
    for (int i=0; i<glp_cols-1; i++) {
      int k = nextpow2(varExplainedSum[i] / varExplainedSum[i+1]);
      if (std::isnan(k) || k > 10 || k < 0) {
        k = 0;
      }
      
      glp_set_row_bnds(lp, rowCounter, GLP_UP, 0.0, k);

      for (int j=0; j<glp_cols; j++) {
        rowIndices[lastMatIdx] = rowCounter;
        colIndices[lastMatIdx] = j+1;
        if (i == j) {
          numVal[lastMatIdx] = 1;
        } else if (i+1 == j) {
          numVal[lastMatIdx] = -1;
        }
        lastMatIdx++;
      }
      
      rowCounter++;
    }

    glp_load_matrix(lp, lastMatIdx-1, rowIndices.data(), colIndices.data(), numVal.data());
    // int ret = glp_simplex(lp, NULL);
    int ret = glp_simplex(lp, &parm);
    // int ret = glp_exact(lp, &parm);
    if (ret != 0) {
      std::cout << "glp solver failed: " << ret << std::endl;
    }

    // std::cout << "objective function value: " << glp_get_obj_val(lp) << std::endl;

    std::vector<int> roundUpCandidate;
    roundUpCandidate.reserve(glp_cols);

    std::vector<int> solutionX(glp_cols, 0);
    // std::cout << "glp solution: " << std::endl;
    int totalBit = 0;
    for (int i=1; i<=glp_cols; i++) {
      if (false) {  // change to 'true' to use 1 uniform bit allocation
        solutionX[i-1] = 1;
        
        totalBit += solutionX[i-1];
        if (i == bitBudget) {
          break;
        }
      } else {
        solutionX[i-1] = (int)std::trunc(glp_get_col_prim(lp, i));
        totalBit += solutionX[i-1];

        if (std::trunc(glp_get_col_prim(lp, i)) != glp_get_col_prim(lp, i)) {
          roundUpCandidate.push_back(i);
        }
      }
      
      // std::cout << glp_get_col_prim(lp, i) << ",";
    }
    // std::cout << std::endl;
    // std::cout << "totalBit: " << totalBit << std::endl;

    if (totalBit < bitBudget) {
      // round up first
      if (roundUpCandidate.size() > 0) {
        int it = 0;
        while (totalBit < bitBudget) {
          solutionX[roundUpCandidate[it]-1] += 1;
          totalBit++;
          it++;
          if (it >= (int)roundUpCandidate.size()) {
            break;
          }
        }
      }
      
      // still totalBit < bitBudget
      if (totalBit < bitBudget) {
        for (int i=0; i<(int)solutionX.size(); i++) {
          if (solutionX[i] < 8) {
            solutionX[i] += 1;
            totalBit++;
            if (totalBit >= bitBudget) {
              break;
            }
          }
        }
      }
    }
    // std::cout << "solutionX:" << std::endl;
    // for (int i=0; i<(int)solutionX.size(); i++) {
    //   std::cout << solutionX[i] << ",";
    // }
    // std::cout << std::endl;
    glp_delete_prob(lp);

    auto quantile = [](const std::vector<float> &Z, int N) -> std::vector<float> {
      std::vector<float> ret(N);
      float p;
      for (int i=0; i<N; i++) {
        p = (float)(i+1)/(N+1);
        
        // matlab equivalent method - source: https://stackoverflow.com/a/37708864
        float poi = (1 - p)*(-0.5) + p * ((float)Z.size() - 0.5);
        size_t left = std::max((int64_t)std::floor(poi), (int64_t)0);
        size_t right = std::min((int64_t)std::ceil(poi), (int64_t)Z.size() - 1);
        ret[i] = (1 - (poi-left))*Z[left] + (poi - left)*Z[right];
      }
      
      return ret;
    };

    auto cumsumSolutionX = [](const std::vector<int> &Z) -> std::vector<int> {
      std::vector<int> cumSumVar(Z.size() + 1);
      cumSumVar[0] = 0;
      cumSumVar[1] = Z[0];
      for (int i=1; i<(int)Z.size(); i++) {
        cumSumVar[i+1] = Z[i] + cumSumVar[i];
      }

      return cumSumVar;
    };

    // std::cout << "bitPosition:" << std::endl;
    std::vector<int> bitPosition = cumsumSolutionX(solutionX);
    // for (auto bp: bitPosition) {
    //   std::cout << bp << ", ";
    // }
    // std::cout << std::endl;

    Eigen::MatrixXf Q(256, glp_cols);
    for (int i=0; i<glp_cols; i++) {
      if (solutionX[i] > 0) {
        int N = std::pow(2, solutionX[i]) - 1;
        
        // put xtrain col to sorted vector
        std::vector<float> sortedVec(XTrainChecked.rows());
        for (int j=0; j<XTrainChecked.rows(); j++) {
          sortedVec[j] = XTrainChecked(j, i);
        }
        std::sort(sortedVec.begin(), sortedVec.end());

        std::vector<float> quantileTmp = quantile(sortedVec, N);
        for (int j=0; j<(int)quantileTmp.size(); j++) {
          Q(j, i) = quantileTmp[j];
        }

        Q(quantileTmp.size(), i) = std::numeric_limits<float>::max(); // fill last row with max value
      }
    }

    auto encodeToBinary = [&solutionX, &bitPosition, &Q](const Eigen::MatrixXf &X, bitvectors &out) {
      for (int i=0; i<X.rows(); i++) {
        for (int j=0; j<Q.cols(); j++) {
          if (solutionX[j] > 0) {
            int qMax = 1 << solutionX[j]; // 2^solutionX[j]

            for (int q=0; q<qMax; q++) {
              if (X(i, j) <= Q(q, j)) {
                uint64_t bucket = (uint64_t)q;

                int blockIdxStart = bitPosition[j] / 64;
                if ((int)(bitPosition[j] / 64) != (int)((bitPosition[j+1]-1) / 64)) { // sliced
                  int right = solutionX[j] - ((blockIdxStart+1)*64-bitPosition[j]);
                  out[i][blockIdxStart] |= bucket >> right;
                  out[i][blockIdxStart+1] |= (bucket & ((1ull << right) - 1)) << (64-right);
                } else {  // not sliced
                  out[i][blockIdxStart] |= bucket << (64-(bitPosition[j] % 64)-solutionX[j]);
                }
                break;
              }
            }
          }
        }
      }
    };

    encodeToBinary(XTrainChecked, XTrainBinary);
    encodeToBinary(XTestChecked, XTestBinary);
  }
  
  void binaryEncodingLUT(const Eigen::MatrixXf &XTrain, const int bitBudget, CodebookType &codebookOut, bool verbose=false) {
    /* Compute PCA on training data */
    Eigen::EigenSolver<Eigen::MatrixXf> es(XTrain.transpose() * XTrain);
    
    // get sorted index
    std::vector<IdxDistPairComplexFloat> eigvalueidx(es.eigenvalues().rows());
    for (int i = 0; i < es.eigenvalues().rows(); i++) {
      eigvalueidx[i].dist = es.eigenvalues()(i);
      eigvalueidx[i].idx = i;
    }

    std::sort(eigvalueidx.begin(), eigvalueidx.end(),
      [](const IdxDistPairComplexFloat &a, const IdxDistPairComplexFloat &b) -> bool {
        if (a.dist.real() == b.dist.real()) {
          return a.dist.imag() > b.dist.imag();
        }
        return a.dist.real() > b.dist.real();
      }
    );

    this->eigenVectors.resize(es.eigenvectors().rows(), es.eigenvectors().cols());
    for (int i = 0; i < eigenVectors.cols(); i++) {
      eigenVectors.col(i) = es.eigenvectors().col(eigvalueidx[i].idx);
    }

    /* New representation: Project data to learned eigenvectors V2 */
    Eigen::MatrixXf XTrainChecked = this->ProjectOnEigenVectors(XTrain);
    
    /* Assign bits to dimensions */
    Eigen::VectorXf varExplainedSum(eigvalueidx.size());
    for (int i=0; i<(int)eigvalueidx.size(); i++) {
      varExplainedSum(i) = eigvalueidx[i].dist.real();
    }
    varExplainedSum /= varExplainedSum.sum();

    if (verbose) {
      for (int i=0; i<varExplainedSum.rows(); i++) {
        std::cout << varExplainedSum(i) << ", ";
        if (varExplainedSum(i) < 1e-11) {
          varExplainedSum(i) = 1e-17;
        }
      }
      std::cout << std::endl;
    }

    // MILP
    glp_prob *lp = glp_create_prob();
    glp_smcp parm; glp_init_smcp(&parm);
    // parm.meth = GLP_DUAL;
    // parm.pricing = GLP_PT_STD;
    // parm.r_test = GLP_RT_STD;
    // parm.presolve = GLP_ON;
    glp_set_obj_dir(lp, GLP_MAX);
    int glp_rows = varExplainedSum.rows(), glp_cols = varExplainedSum.rows();
    glp_add_rows(lp, glp_rows);
    glp_add_cols(lp, glp_cols);
    
    // set integer constraint
    for (int d=1; d<=glp_cols; d++) {
      glp_set_col_kind(lp, d, GLP_IV);
    }

    for (int i=0; i<glp_cols; i++) {
      glp_set_obj_coef(lp, i+1, varExplainedSum(i));
    }

    auto cumsum = [](const Eigen::VectorXf &Z) -> std::vector<float> {
      std::vector<float> cumSumVar(Z.rows());
      cumSumVar[0] = Z(0);
      for (int i=1; i<Z.rows(); i++) {
        cumSumVar[i] = Z(i) + cumSumVar[i-1];
      }

      return cumSumVar;
    };

    std::vector<float> cumSumVar = cumsum(varExplainedSum);
    float UniformAllocVarExplained = (bitBudget <= (int)cumSumVar.size()) ? cumSumVar[bitBudget-1] : cumSumVar.back();
    float varExplainedXPercentage = 0.99 * UniformAllocVarExplained;

    int lastMatIdx = 1, rowCounter = 1;
    std::vector<int> rowIndices(glp_rows * glp_cols + 1, 0), colIndices(glp_rows * glp_cols + 1, 0);
    std::vector<double> numVal(glp_rows * glp_cols + 1, 0);

    // Bit allocation per dimension 0 or more
    // x_i >= 0 or -x_i <=0
    // x_i <= 8
    for (int i=1; i<=glp_cols; i++) {
      double lb = 0;
      if (i <= bitBudget) {
        lb = (varExplainedXPercentage >= cumSumVar[i-1]) ? 1.0 : 0.0;
      }
      glp_set_col_bnds(lp, i, GLP_DB, lb, 8.0);
    }

    // sum(x_i) = budget
    // 1 1 1 1 1 = budget
    glp_set_row_bnds(lp, rowCounter, GLP_FX, bitBudget, 0.0);
    for (int d = 1; d <= glp_cols; d++) {
      rowIndices[lastMatIdx] = rowCounter;
      colIndices[lastMatIdx] = d;
      numVal[lastMatIdx] = 1.0;
      lastMatIdx++;
    }
    rowCounter++;

    auto nextpow2 = [](double x) -> int {
      if (x == 0) {
        return 0;
      }
      return (int)std::pow(2, std::floor(std::log2(std::abs(x))));
    };
    
    // Allocate bits based on the explained variance
    // 2^(x_i) >= ratioVar 2^(x_i+1) 
    // k = nextpow2(ratioVar)
    // or
    // x_i - x_(i+1) <= k
    for (int i=0; i<glp_cols-1; i++) {
      int k = nextpow2(varExplainedSum[i] / varExplainedSum[i+1]);
      if (std::isnan(k) || k > 10 || k < 0) {
        k = 0;
      }
      glp_set_row_bnds(lp, rowCounter, GLP_UP, 0.0, k);

      for (int j=0; j<glp_cols; j++) {
        rowIndices[lastMatIdx] = rowCounter;
        colIndices[lastMatIdx] = j+1;
        if (i == j) {
          numVal[lastMatIdx] = 1;
        } else if (i+1 == j) {
          numVal[lastMatIdx] = -1;
        }
        lastMatIdx++;
      }
      
      rowCounter++;
    }

    glp_load_matrix(lp, lastMatIdx-1, rowIndices.data(), colIndices.data(), numVal.data());
    // int ret = glp_simplex(lp, NULL);
    int ret = glp_simplex(lp, &parm);
    // int ret = glp_exact(lp, &parm);
    if (ret != 0) {
      std::cout << "glp solver failed: " << ret << std::endl;
    }

    std::vector<int> roundUpCandidate;
    roundUpCandidate.reserve(glp_cols);

    this->solutionX.resize(glp_cols, 0);
    if (verbose) {
      std::cout << "glp solution: " << std::endl;
    }
    int totalBit = 0;
    for (int i=1; i<=glp_cols; i++) {
      if (false) {  // change to 'true' to use 1 uniform bit allocation
        solutionX[i-1] = 1;
        
        totalBit += solutionX[i-1];
        if (totalBit == bitBudget) {
          break;
        }
      } else {
        solutionX[i-1] = (int)std::trunc(glp_get_col_prim(lp, i));
        totalBit += solutionX[i-1];

        if (std::trunc(glp_get_col_prim(lp, i)) != glp_get_col_prim(lp, i)) {
          roundUpCandidate.push_back(i);
        }
      }
      
      if (verbose) {
        std::cout << glp_get_col_prim(lp, i) << ",";
      }
    }
    if (verbose)  {
      std::cout << std::endl;
      std::cout << "totalBit: " << totalBit << std::endl;
    }

    if (totalBit < bitBudget) {
      // round up first
      if (roundUpCandidate.size() > 0) {
        int it = 0;
        while (totalBit < bitBudget) {
          solutionX[roundUpCandidate[it]-1] += 1;
          totalBit++;
          it++;
          if (it >= (int)roundUpCandidate.size()) {
            break;
          }
        }
      }
      
      // still totalBit < bitBudget
      if (totalBit < bitBudget) {
        for (int i=0; i<(int)solutionX.size(); i++) {
          if (solutionX[i] < 8) {
            solutionX[i] += 1;
            totalBit++;
            if (totalBit >= bitBudget) {
              break;
            }
          }
        }
      }
    }
    if (verbose) {
      std::cout << "this->solutionX:" << std::endl;
      for (int i=0; i<(int)solutionX.size(); i++) {
        std::cout << solutionX[i] << ",";
      }
      std::cout << std::endl;
    }
    glp_delete_prob(lp);

    auto centroidsQuantile = [](const std::vector<float> &Z, int N, std::vector<float> &quantiles, float *centroids) {
      quantiles.front() = Z.front();
      float p;
      for (int i=0; i<N-1; i++) {
        p = (float)(i+1)/N;
        
        // matlab equivalent method - source: https://stackoverflow.com/a/37708864
        float poi = (1 - p)*(-0.5) + p * ((float)Z.size() - 0.5);
        size_t left = std::max((int64_t)std::floor(poi), (int64_t)0);
        size_t right = std::min((int64_t)std::ceil(poi), (int64_t)Z.size() - 1);
        quantiles[i+1] = (1 - (poi-left))*Z[left] + (poi - left)*Z[right];
      }
      quantiles.back() = Z.back();
      
      int lastidx = 0;
      for (int i=0; i<N; i++) {
        int count = 0;
        while (lastidx < (int)Z.size() && Z[lastidx] <= quantiles[i+1]) {
          centroids[i] += Z[lastidx];
          lastidx++;
          count++;
        }

        if (count > 0) {
          centroids[i] /= count;
        } else {  // if there is no value inside the bucket, use median instead
          centroids[i] = (quantiles[i] + quantiles[i+1]) / 2.0f;
        }
      }
    };

    // count non-zero allocation
    this->nonZeroAllocCount = 0;
    for (int i=0; i<glp_cols; i++) {
      if (solutionX[i] > 0) {
        nonZeroAllocCount += 1;
      } else {
        break;
      }
    }

    this->centroidsMat.resize(256, nonZeroAllocCount);
    centroidsMat.setZero();
    Matrixf Q(nonZeroAllocCount, std::vector<float>());
    for (int i=0; i<nonZeroAllocCount; i++) {
      int N = std::pow(2, solutionX[i]);
      
      // put xtrain col to sorted vector
      std::vector<float> sortedVec(XTrainChecked.rows());
      for (int j=0; j<XTrainChecked.rows(); j++) {
        sortedVec[j] = XTrainChecked(j, i);
      }
      std::sort(sortedVec.begin(), sortedVec.end());

      Q[i].resize(N+1);
      centroidsQuantile(sortedVec, N, Q[i], centroidsMat.col(i).data());
    }

    auto cumsumSolutionX = [](const std::vector<int> &Z) -> std::vector<int> {
      std::vector<int> cumSumVar(Z.size() + 1);
      cumSumVar[0] = 0;
      cumSumVar[1] = Z[0];
      for (int i=1; i<(int)Z.size(); i++) {
        cumSumVar[i+1] = Z[i] + cumSumVar[i];
      }

      return cumSumVar;
    };

    this->LUTBitPosition = cumsumSolutionX(solutionX);
    if (verbose) {
      std::cout << "bitPosition:" << std::endl;
      for (auto bp: LUTBitPosition) {
        std::cout << bp << ", ";
      }
      std::cout << std::endl;
    }

    auto encodeToLUTCode = [&Q, this](const Eigen::MatrixXf &X, CodebookType &codebook) {
      codebook.resize(X.rows(), nonZeroAllocCount);
      codebook.setZero();
      for (int i=0; i<X.rows(); i++) {
        for (int j=0; j<(int)Q.size(); j++) {
          int qMax = (1 << solutionX[j]) + 1; // 2^solutionX[j]

          for (uint16_t q=0; q<(uint16_t)qMax; q++) {
            if (X(i, j) <= Q[j][q]) {
              // assign which centroids
              uint16_t code;
              float l, m, r;
              if (q == 0) {
                code = 0;
              } else if (q == 1) {
                m = std::abs(X(i, j) - centroidsMat(q-1, j));
                r = std::abs(X(i, j) - centroidsMat(q, j));
                code = (m <= r) ? (q-1) : q;
              } else if (q == qMax-1) {
                m = std::abs(X(i, j) - centroidsMat(q-1, j));
                l = std::abs(X(i, j) - centroidsMat(q-2, j));
                code = (m <= l) ? (q-1) : (q-2);
              } else {
                m = std::abs(X(i, j) - centroidsMat(q-1, j));
                l = std::abs(X(i, j) - centroidsMat(q-2, j));
                r = std::abs(X(i, j) - centroidsMat(q, j));
                if (m <= l && m <= r) {
                  code = q-1;
                } else if (l <= m && l <= r) {
                  code = q-2;
                } else {
                  code = q;
                }
              }

              codebook(i, j) = (uint8_t)code;
              break;
            } else if (q == qMax-1) {
              codebook(i, j) = (uint8_t)(q-1);
            }
          }
        } 
      }
    };

    encodeToLUTCode(XTrainChecked, codebookOut);
  }

  void computeTIClusters(const CentroidsPerSubsType &centroidsPerDim, CodebookType &codebook, const int clusterNum) {
    int totalDimension = 0;
    for (int i=0; i<(int)this->subspacesBeginIdx.size()-1; i++) {
      totalDimension += subspacesBeginIdx[i+1] - subspacesBeginIdx[i];
    }

    this->clustersVAQ.resize(clusterNum, totalDimension);
    clustersVAQ.setZero();

    // randomly create cluster from dataset
    if (true) {
      for (int i=0; i < (int)clustersVAQ.rows(); i++) {
        int randIdx = rand() % codebook.rows();
        for (int subvec=0; subvec < (int)centroidsPerDim.size(); subvec++) {
          clustersVAQ.row(i).segment(subspacesBeginIdx[subvec], subspacesBeginIdx[subvec+1]-subspacesBeginIdx[subvec]) = centroidsPerDim[subvec].row(codebook(randIdx, subvec));
        }
      }
    } else {  // use kmeans
      clustersVAQ = KMeans::staticFitCodebook(codebook, centroidsPerDim, centroidsPerDim.size(), clusterNum, 25, true);
    }

    // compute distance from each cluster centroids to dictionary
    std::vector<std::vector<std::vector<float>>> ccToDictDist(clustersVAQ.rows(), std::vector<std::vector<float>>(nonZeroAllocCount));
    for (int z=0; z<(int)clustersVAQ.rows(); z++) {
      for (int i=0; i<nonZeroAllocCount; i++) {
        ccToDictDist[z][i].resize(centroidsPerDim[i].rows(), 0);
        for (int j=0; j<(int)centroidsPerDim[i].rows(); j++) {
          ccToDictDist[z][i][j] = (centroidsPerDim[i].row(j) - clustersVAQ.row(z).segment(subspacesBeginIdx[i], subspacesBeginIdx[i+1]-subspacesBeginIdx[i])).squaredNorm();
        }
      }
    }

    // assign each vector to nearest clusters and save the distance
    this->codeToCCDist.reserve(codebook.rows());
    this->clusterMembersVAQ.resize(clustersVAQ.rows());
    for (int i=0; i<(int)clustersVAQ.rows(); i++) {
      clusterMembersVAQ[i].reserve(codebook.rows() / clusterNum * 2);
    }
    auto codes = codebook.data();
    for (int i=0; i<(int)codebook.rows(); i++) {
      float closestDist = std::numeric_limits<float>::max();
      int closestIdx = 0;
      for (int z=0; z<(int)clustersVAQ.rows(); z++) {
        float dist = 0;
        for (int col=0; col<nonZeroAllocCount; col++) {
          dist += ccToDictDist[z][col][codes[col]];
        }
        dist = std::sqrt(dist);
        if (dist < closestDist) {
          closestDist = dist;
          closestIdx = z;
        }
      }
      codeToCCDist.push_back(closestDist);
      clusterMembersVAQ[closestIdx].push_back(i);

      codes += codebook.cols();
    }

    // sort clusterMembersVAQ from the farthest to centroid
    for (auto &cm: clusterMembersVAQ) {
      std::sort(cm.begin(), cm.end(),
        [this](int i, int j) {
          return this->codeToCCDist[i] > this->codeToCCDist[j];
        }
      );
    }

    // sort group the codebook
    this->clusterMembersStartIdx.resize(clusterMembersVAQ.size());
    CodebookType groupedCodebook(codebook.rows(), codebook.cols());
    int clusterMemberIdx = 0;
    int rowCounter = 0;
    for (auto &cm: clusterMembersVAQ) {
      clusterMembersStartIdx[clusterMemberIdx] = rowCounter;
      for (const int idx: cm) {
        groupedCodebook.row(rowCounter) = codebook.row(idx);
        rowCounter++;
      }
      clusterMemberIdx++;
    }
    codebook = groupedCodebook;
  }

  /**
   * @brief Store bitvectors
   * 
   * @param bitVectors bitvectors to be loaded
   */
  void loadBitV(bitvectors bitVectors);
  
  /**
   * @brief Store original dataset
   * 
   * @param oriDataset original dataset
   */
  void loadOriginal(Matrixf oriDataset);

  
  void computeTriangleInequalityClusters() {
    const int nrows = (int)originalData.size();
    const int ncols = (int)originalData.at(0).size();
    std::cout << nrows << " " << ncols << std::endl;

    // assign each vector to nearest clusters and save the distance
    datapointToCCDist.reserve(nrows);
    clusterMembers.resize(this->centroids_size);
    for (int i=0; i<centroids_size; i++) {
      clusterMembers[i].reserve(nrows / centroids_size * 2);
    }
    for (int i=0; i<(int)nrows; i++) {
      float closestDist = std::numeric_limits<float>::max();
      int closestIdx = 0;
      for (int centIdx=0; centIdx<centroids_size; centIdx++) {
        float dist = euclideanDist(originalData[i], this->centroids[centIdx]);
        if (dist < closestDist) {
          closestDist = dist;
          closestIdx = centIdx;
        }
      }

      datapointToCCDist.push_back(closestDist);
      clusterMembers[closestIdx].push_back(i);
    }

    // sort clusterMembers from the farthest to centroid
    for (auto &cm: clusterMembers) {
      std::sort(cm.begin(), cm.end(), 
        [this](int i, int j) {
          return this->datapointToCCDist[i] > this->datapointToCCDist[j];
        }
      );
    }

    // sort group the datapoint
    this->clusterMembersStartIdx.resize(clusterMembers.size());
    Matrixf groupedDataset(nrows);
    for (int i=0; i<nrows; i++) {
      groupedDataset[i].reserve(ncols);
    }
    int clusterMemberIdx = 0;
    int rowCounter = 0;
    for (auto &cm: clusterMembers) {
      clusterMembersStartIdx[clusterMemberIdx] = rowCounter;
      for (const int idx: cm) {
        groupedDataset[rowCounter] = originalData[idx];
        rowCounter++;
      }
      clusterMemberIdx++;
    }
    originalData = groupedDataset;
  }

  /**
   * @brief Store centroids in binary form
   * 
   * @param _centroids 
   */
  void loadCentroidsBin(bitvectors _centroids);
  
  /**
   * @brief Store centroids in original form
   * 
   * @param _centroids 
   */
  void loadCentroids(Matrixf _centroids);
  
  /**
   * @brief Store cluster info
   * 
   * @param clustIdx 
   * @param centroids_size
   */
  void loadClusterInfo(std::vector<int> clustIdx, int _centroids_size);
  

  /**
   * @brief Query k-nearest-neigbor
   * 
   * @param queries query points
   * @param k number of neighbor
   * @param method query method (SORT or HEAP)
   * @return std::vector<std::vector<IdxDistPair>> List of List of (idx, distance) nearest neighbor, sorted
   */
  std::vector<std::vector<IdxDistPair>> query(const bitvectors &queries, const int k, int method=QueryMethod::Sort) const;
  
  /**
   * @brief Query k-nearest-neighbor with combination of euclidean distance
   * 
   * @param queries query instances
   * @param oriQueries query instances in original form
   * @param k number of neighbor
   * @param method query method (SORT or HEAP)
   * @return std::vector<std::vector<IdxDistPair>> List of List of (idx, distance) nearest neighbor, sorted
   */
  std::vector<std::vector<IdxDistPairFloat>> queryRerank(const bitvectors &queries, const Matrixf &oriQueries, const int k, const int factor=2, int method=QueryMethod::Sort, bool earlyAbandonRank=false) const;
  

  /**
   * @brief Query k-nearest-neighbor with cluster information
   * 
   * @param queries query instances
   * @param clustIdx 
   * @param centroids 
   * @param k number of neighbor
   * @param method query method (SORT or HEAP)
   * @return std::vector<std::vector<IdxDistPair>> 
   */
  std::vector<std::vector<IdxDistPair>> queryWithClusterInfo(const bitvectors &queries, const Matrixf &oriQueries, const int k, int method=QueryMethod::Sort, int atleastCluster=1) const;

  /**
   * @brief Query k-nearest-neighbor with combination of euclidean distance and cluster information
   * 
   * @param queries query instances
   * @param oriQueries query instances in original form
   * @param centroids 
   * @param k number of neighbor
   * @param method query method (SORT or HEAP)
   * @return std::vector<std::vector<IdxDistPairFloat>> 
   */
  std::vector<std::vector<IdxDistPairFloat>> queryRerankWithClusterInfo(const bitvectors &queries, const Matrixf &oriQueries, const int k, const int factor, int method=QueryMethod::Sort, int atleastCluster=1, bool earlyAbandonRank=false) const;
  
  /**
   * @brief 
   * 
   * @param oriQueries 
   * @param k 
   * @param method 
   * @return std::vector<std::vector<IdxDistPairFloat>> 
   */
  std::vector<std::vector<IdxDistPairFloat>> queryNaive(const Matrixf &oriQueries, const int k, int method=QueryMethod::Sort) const;
  std::vector<std::vector<IdxDistPairFloat>> queryNaiveWithClusterInfo(const Matrixf &oriQueries, const int k, int method=QueryMethod::Sort, int atleastCluster=1) const;
  std::vector<std::vector<IdxDistPairFloat>> queryNaiveWithClusterInfoParallelDiskResident(const std::string datasetFilepath, const Matrixf &oriQueries, const std::vector<int> &clustIdx, const int k, int thread, int method=QueryMethod::Sort, int batch=100000000, int atleastCluster=1) const;
  std::vector<std::vector<IdxDistPairFloat>> queryNaiveTriangleInequality(const Matrixf &oriQueries, const int k, int method) const;
  template<typename T>
  std::vector<std::vector<IdxDistPairFloat>> queryNaiveEigen(const Eigen::MatrixBase<T> &dataset, const Eigen::MatrixBase<T> &queries, const int k) {
    const int dataNRows = dataset.rows();
    std::vector<std::vector<IdxDistPairFloat>> answers(queries.rows());

    for (int q_idx=0; q_idx < queries.rows(); q_idx++) {
      std::vector<float> pairsDist(dataNRows);
      
      for (int dataIndex = 0; dataIndex < dataNRows; dataIndex++) {
        pairsDist[dataIndex] = (queries.row(q_idx) - dataset.row(dataIndex)).squaredNorm();
      }

      answers[q_idx] = BitVecEngine::KNNFromDists(pairsDist.data(), dataNRows, k);
    }
    
    return answers;
  }

  /**
   * @brief Query k-nearest-neigbor with Progressive Filtering with sort method
   * 
   * @param queries query points
   * @param k number of neighbor
   * @param selectivity
   * @return std::vector<std::vector<IdxDistPair>> List of List of (idx, distance) nearest neighbor, sorted
   */
  std::vector<std::vector<IdxDistPair>> queryFiltering_Sort(const bitvectors &queries, const int k) const;


  /**
   * @brief Query k-nearest-neigbor with Progressive Filtering with heap method
   * 
   * @param queries query points
   * @param k number of neighbor
   * @return std::vector<std::vector<IdxSubDistPair<SubVector>>> 
   */
  std::vector<std::vector<IdxSubDistPair>> queryFiltering_Heap(const bitvectors &queries, const int k) const;

  /**
   * @brief Query k-nearest-neigbor in parallel manner
   * 
   * @param queries query points
   * @param k number of neighbor
   * @param distanceFunction 
   * @param thread Number of thread
   * @return std::vector<std::vector<IdxDistPair>> List of List of (idx, distance) nearest neighbor, sorted
   */
  std::vector<std::vector<IdxDistPair>> queryParallel(const bitvectors &queries, const int k, const int thread) const;
  std::vector<std::vector<IdxDistPairFloat>> queryNaiveParallel(const Matrixf &queries, const int k, const int thread, int method=QueryMethod::Sort) const;
  std::vector<std::vector<IdxDistPairFloat>> queryNaiveParallelDiskResident(const std::string datasetFilepath, const Matrixf &queries, const int k, const int thread, int method=QueryMethod::Sort, int batch=100000000) const;

  std::vector<std::vector<IdxDistPairFloat>> queryLUT(const Eigen::MatrixXf &queries, const int k, const CodebookType &codebook) {
    const int dataNRows = codebook.rows();
    std::vector<std::vector<IdxDistPairFloat>> answers(queries.rows());

    Eigen::MatrixXf queriesPCA = this->ProjectOnEigenVectors(queries, true);
    // std::cout << "queriesPCA: " << std::endl;
    // std::cout << queriesPCA << std::endl;

    Eigen::Matrix<float, 256, Eigen::Dynamic, Eigen::ColMajor> lut(256, nonZeroAllocCount);
    
    for (int q_idx=0; q_idx < (int)queriesPCA.rows(); q_idx++) {
      auto createLUT = [this, &queriesPCA, &lut, q_idx]() {

        static constexpr int packet_width = 8; // objs per simd register
        for (int dimen=0; dimen<nonZeroAllocCount; dimen++) {
          if (this->solutionX[dimen] >= 3) { // only vectorized when centroids >= 8
            const int nstripes = (int)(std::ceil(std::pow(2, solutionX[dimen]) / packet_width));
            __m256 accumulators[256/8];  // max centroids
            
            auto lut_ptr = lut.data() + lut.rows()*dimen;
            auto centroids_ptr = centroidsMat.data() + centroidsMat.rows()*dimen;

            for (int i=0; i<nstripes; i++) {
              accumulators[i] = _mm256_setzero_ps();
            }

            auto q_broadcast = _mm256_set1_ps(queriesPCA(q_idx, dimen));
            for (int i=0; i<nstripes; i++) {
              auto centroids_col = _mm256_load_ps(centroids_ptr);
              centroids_ptr += packet_width;

              auto diff = _mm256_sub_ps(q_broadcast, centroids_col);
              accumulators[i] = fma(diff, diff, accumulators[i]);
            }

            // write out dists in this col of the lut
            for (uint16_t i=0; i<nstripes; i++) {
              _mm256_store_ps((float *)lut_ptr, accumulators[i]);
              lut_ptr += packet_width;
            }
          } else {
            const int ncentroids = 1 << solutionX[dimen];
            for (int cIdx=0; cIdx < ncentroids; cIdx++) {
              auto diff = queriesPCA(q_idx, dimen) - centroidsMat(cIdx, dimen);
              lut(cIdx, dimen) = diff*diff;
            }
          }
        }
      };

      // create lookup-tables distance to centroids
      lut.setZero();
      createLUT();

      // std::cout << "LUT contains: " << std::endl;
      // std::cout << lut << std::endl;

      // std::cout << "codebook contains: " << std::endl;
      // std::cout << codebook.cast<int>() << std::endl;

      if (true) { // early abandon
        auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
          return a.dist < b.dist;
        };

        std::vector<IdxDistPairFloat> pairs;
        pairs.reserve(k+1);
        std::make_heap(pairs.begin(), pairs.end(), comparator);
        float bsfK = std::numeric_limits<float>::max();
        float dist;
        
        auto codes = codebook.data();
        for (int dataIndex = 0; dataIndex < (int)codebook.rows(); dataIndex++) {
          dist = 0;
          for (int col=0; col<nonZeroAllocCount && dist < bsfK; col++) {
            auto luts = lut.data() + 256 * col;
            // pairsDist[dataIndex] += luts[codes[col]];
            dist += luts[codes[col]];
          }
          if (dist < bsfK) {
            pairs.emplace_back(dataIndex, dist);
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            if (dataIndex >= k) {
              std::pop_heap(pairs.begin(), pairs.end(), comparator);
              pairs.pop_back();
              bsfK = (pairs.front()).dist;
            } else if (dist > bsfK) { // dataIndex < k
              bsfK = dist;
            }
          }

          codes += codebook.cols();
        }

        std::sort_heap(pairs.begin(), pairs.end(), comparator);
        answers[q_idx] = std::move(pairs);
      } else if (false) { // using annulus ring
        
      } else {
        std::vector<float> pairsDist(dataNRows, 0);
        auto codes = codebook.data();
        for (int dataIndex = 0; dataIndex < dataNRows; dataIndex++) {
          for (int col=0; col<nonZeroAllocCount; col++) {
            auto luts = lut.data() + 256 * col;
            pairsDist[dataIndex] += luts[codes[col]];
          }

          codes += codebook.cols();
        }
        
        answers[q_idx] = KNNFromDists(pairsDist.data(), pairsDist.size(), k);
      }

      // std::cout << "answers[q_idx]: " << std::endl;
      // for (auto &a: answers[q_idx]) {
      //   std::cout << "(" << a.idx << ", " << a.dist << "), ";
      // }
      // std::cout << std::endl;
    }

    return answers;
  }

  std::vector<std::vector<IdxDistPairFloat>> refineAnswer(const Eigen::MatrixXf &queries, const std::vector<std::vector<IdxDistPairFloat>> &answersIn, const Eigen::MatrixXf &dataset, const int k) {
    int refineNum = answersIn.at(0).size();
    std::vector<std::vector<IdxDistPairFloat>> answers(queries.rows());
    auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
      return a.dist < b.dist;
    };
    
    if (true) {
      // heap
      for (int q_idx=0; q_idx < (int)queries.rows(); q_idx++) {
        std::vector<IdxDistPairFloat> pairs;
        pairs.reserve(k+1);
        std::make_heap(pairs.begin(), pairs.end(), comparator);

        for (int i=0; i<refineNum; i++) {
          pairs.push_back(IdxDistPairFloat(
            answersIn[q_idx][i].idx,
            (queries.row(q_idx) - dataset.row(answersIn[q_idx][i].idx)).squaredNorm()
          ));
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          if (i+1 > k) {
            std::pop_heap(pairs.begin(), pairs.end(), comparator);
            pairs.pop_back();
          }
        }
        std::sort_heap(pairs.begin(), pairs.end(), comparator);
        answers[q_idx] = std::move(pairs);
      }
    } else {
      // sort
      for (int q_idx=0; q_idx < (int)queries.rows(); q_idx++) {
        std::vector<float> dists(refineNum);
        std::vector<int> indices(refineNum);
        for (int i=0; i < refineNum; i++) {
          indices[i] = answersIn[q_idx][i].idx;
          dists[i] = (queries.row(q_idx) - dataset.row(answersIn[q_idx][i].idx)).squaredNorm();
        }
        answers[q_idx] = KNNFromDistsIndicesSupplied(dists.data(), indices.data(), refineNum, k);
      }
    }

    return answers;
  }
  
  /**
   * @brief Get the bitv object from data
   * 
   * @param idx bitv index in data
   * @return bitv
   */
  bitv getBitV(const int idx);

  /**
   * @brief Append bitvectors to data
   * 
   * @param bitVectors
   */
  void appendBitV(const bitvectors &bitVectors);

  /**
   * @brief Delete bitv given index
   * 
   * @param idx index
   */
  void deleteBitV(const int idx);

  /**
   * @brief print out data content
   * 
   */
  void printBitVectors() {
    printf("Bit vectors content:\n");
    for (auto it=data.begin(); it != data.end(); it++) {
      for (int i=0; i<(int)(*it).size(); i++) {
        printf("%lu ", (*it)[i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  /**
   * @brief Generate random bitvectors
   * 
   * @param bv output variable
   * @param size size
   * @param randomState
   */
  static void generateDummyBitVectors(const int N, bitvectors &bv, const int size, const int randomState = -1) {
    srand((randomState == -1) ? time(0) : randomState);
    const uint64_t mask = LSB(N <= 64 ? N : 64);

    for (int i=0; i<size; i++) {
      bitv num(actualBitVLen(N), 0);
      if (N <= 32) {
        num[0] = (uint64_t)rand() & mask;
      } else if (N <= 64) {
        num[0] = ((uint64_t)rand() | (((uint64_t)rand()) << 32)) & mask;
      } else {
        const int actualLength = num.size();
        for (int j=0; j<actualLength; j++) {
          num[j] = ((uint64_t)rand() | (((uint64_t)rand()) << 32)) & mask;
        }

        if (actualLength * 64 > N) {
          num[actualLength-1] &= MSB(N - (actualLength-1) * 64);
        }
      }
      bv.push_back(num);
    }
  }
};


#endif  // BITVECENGINE_H_
