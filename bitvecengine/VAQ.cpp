#include <algorithm>
#include <utility>
#include <armadillo>

#include "VAQ.hpp"
#include "utils/TimingUtils.hpp"

/**
 * Core Functions
 */
void VAQ::train(RowMatrixXf &XTrain, bool verbose) {
  START_TIMING(PCA);
  std::cout << XTrain.rows() << " " << XTrain.cols() << std::endl;
  RowMatrixXf covmat(XTrain.cols(), XTrain.cols());

  const int bs = 256 * 1024;
  const int covmatsamplesize = 1000 * XTrain.cols();
  if (covmatsamplesize < XTrain.rows())  {
    RowMatrixXf SampledXTrain(covmatsamplesize, XTrain.cols());
    std::vector<int> perm(XTrain.rows());
    randomPermutation(perm);
    for (int i=0; i<covmatsamplesize; i++) {
      SampledXTrain.row(i) = XTrain.row(perm[i]);
    }
    // process by blocks to avoid using too much RAM
    if (covmatsamplesize > bs) {
      int batchNum = covmatsamplesize / bs;
      if (covmatsamplesize % bs > 0) {
        batchNum += 1;
      }
      covmat.setZero();
      for (int b=0; b<batchNum; b++) {
        int currBatchRows = bs;
        if ((b == batchNum-1) && (covmatsamplesize % bs > 0)) {
          currBatchRows = covmatsamplesize % bs;
        }
        covmat.noalias() += (SampledXTrain.block(b*bs, 0, currBatchRows, SampledXTrain.cols()).transpose() * SampledXTrain.block(b*bs, 0, currBatchRows, SampledXTrain.cols()));
      }
    } else {
      covmat.noalias() = SampledXTrain.transpose() * SampledXTrain;
    }
  } else {
    if (bs < XTrain.rows()) {
      int batchNum = XTrain.rows() / bs;
      if (XTrain.rows() % bs > 0) {
        batchNum += 1;
      }
      covmat.setZero();
      for (int b=0; b<batchNum; b++) {
        int currBatchRows = bs;
        if ((b == batchNum-1) && (XTrain.rows() % bs > 0)) {
          currBatchRows = XTrain.rows() % bs;
        }
        covmat.noalias() += (XTrain.block(b*bs, 0, currBatchRows, XTrain.cols()).transpose() * XTrain.block(b*bs, 0, currBatchRows, XTrain.cols()));
      }
    } else {
      covmat.noalias() = XTrain.transpose() * XTrain;
    }
  }

  Eigen::EigenSolver<RowMatrixXf> es(covmat);
  END_TIMING_V(PCA, "== PCA computation time: ", verbose);

  if (es.info() != Eigen::ComputationInfo::Success) {
    std::cout << "Eigen solver error" << std::endl;
    std::cout << "code: ";
    if (es.info() == Eigen::ComputationInfo::NumericalIssue) {
      std::cout << "Numerical Issue";
    } else if (es.info() == Eigen::ComputationInfo::NoConvergence) {
      std::cout << "No Convergence";
    } else if (es.info() == Eigen::ComputationInfo::InvalidInput) {
      std::cout << "Invalid Input";
    } else {
      std::cout << "Unknown";
    }
    std::cout << std::endl;
    assert(false);
  }

  typedef struct {
    std::complex<float> value;
    int idx;
  } EigValueIdxPair;
  std::vector<EigValueIdxPair> eigvalueidx(es.eigenvalues().rows());
  for (int i = 0; i < es.eigenvalues().rows(); i++) {
    eigvalueidx[i].value = es.eigenvalues()(i);
    eigvalueidx[i].idx = i;
  }
  std::sort(eigvalueidx.begin(), eigvalueidx.end(),
    [](const EigValueIdxPair &a, const EigValueIdxPair &b) -> bool {
      if (a.value.real() == b.value.real()) {
        return a.value.imag() > b.value.imag();
      }
      return a.value.real() > b.value.real();
    }
  );
  mEigenVectors.resize(es.eigenvectors().rows(), es.eigenvectors().cols());
  for (int i = 0; i < mEigenVectors.cols(); i++) {
    mEigenVectors.col(i) = es.eigenvectors().col(eigvalueidx[i].idx);
  }

  const int oriTotalDim = XTrain.cols();
  mSubsLen = oriTotalDim / mSubspaceNum;
  if (oriTotalDim % mSubspaceNum > 0) {
    mSubsLen += 1;
  }

  // Code for next project
  #if 0
  if (mMethods & NNMethod::TI) {
    if (mTIVariance < 1) {
      Eigen::VectorXf varPerDim(eigvalueidx.size());
      for (int i=0; i<(int)eigvalueidx.size(); i++) {
        varPerDim(i) = eigvalueidx[i].value.real();
      }
      varPerDim /= varPerDim.sum();
      Eigen::VectorXf varPerSubs(mSubspaceNum);
      for (int i=0; i<mSubspaceNum; i++) {
        varPerSubs(i) = varPerDim.segment(
          i*mSubsLen, mSubsLen
        ).sum();
      }
      Eigen::VectorXf cumSumVarPerSubs = cumSum(varPerSubs);

      mTISegmentNum = 0;
      for (int i=0; i<mSubspaceNum; i++) {
        if (cumSumVarPerSubs[i] <= mTIVariance)  {
          mTISegmentNum += 1;
        }
      }
      if (mTISegmentNum == 0) {
        mTISegmentNum = 1;
      }
    } else if (mTISegmentNum == -1) {
      mTISegmentNum = mSubspaceNum;
    }

    if (false) {
      mTIClusters.resize(mTIClusterNum, XTrain.cols());
      mTIClusters.setZero();
      // random
      for (int i=0; i<mTIClusterNum; i++) {
        int randIdx = rand() % XTrain.rows();
        mTIClusters.row(i) = XTrain.row(randIdx);
      }
      this->ProjectOnFirstDimsEigenVectorsInPlace(mTIClusters, mTISegmentNum * mSubsLen);
    } else {
      // kmeans before project
      mTIClusters = KMeans::staticFitSampling(
        XTrain, mTIClusterNum, 50, true
      );
      this->ProjectOnFirstDimsEigenVectorsInPlace(mTIClusters, mTISegmentNum * mSubsLen);

      // kmeans after project
      // std::vector<int> perm(XTrain.rows());
      // randomPermutation(perm);
      // int sampleSize = std::max(256 * mTIClusterNum, 256 * 256);
      // sampleSize = std::min((int)XTrain.rows(), sampleSize);
      // RowMatrixXf tempX(sampleSize, XTrain.cols());
      // for (int i=0; i<sampleSize; i++) {
      //   tempX.row(i) = XTrain.row(perm[i]);
      // }
      // std::cout << "santuy" << std::endl;
      // this->ProjectOnFirstDimsEigenVectorsInPlace(tempX, mTISegmentNum * mSubsLen);
      // std::cout << "sini error" << std::endl;
      // mTIClusters = KMeans::staticFitSampling(
      //   tempX, mTIClusterNum, 50, true
      // );
    }

    std::cout << "ti cluster created" << std::endl;

    mCodeToCCDist.resize(XTrain.rows());
    mTIClustersMember.resize(mTIClusterNum);
    for (int i=0; i<(int)mTIClusters.rows(); i++) {
      mTIClustersMember[i].reserve(XTrain.rows() / mTIClusterNum);
    }
    constexpr int threadnum = 1;
    #pragma omp parallel num_threads(threadnum)
    {
      std::vector<std::vector<int>> privateTIClustersMember(mTIClusterNum);
      for (int i=0; i<mTIClusterNum; i++) {
        privateTIClustersMember[i].reserve(XTrain.rows() / (mTIClusterNum*threadnum));
      }

      #pragma omp for nowait schedule(static)
      for (int i=0; i<XTrain.rows(); i++) {
        RowVectorXf x = (XTrain.row(i) * mEigenVectors.block(0, 0, mEigenVectors.rows(), mTISegmentNum * mSubsLen)).real();

        float closestDist = std::numeric_limits<float>::max();
        int closestIdx = -1;
        for (int c=0; c<mTIClusterNum; c++) {
          float dist = (x - mTIClusters.row(c)).norm();
          if (dist < closestDist) {
            closestIdx = c;
            closestDist = dist;
          }
        }
        mCodeToCCDist.at(i) = closestDist;
        privateTIClustersMember[closestIdx].push_back(i);
      }

      const int num_threads = omp_get_num_threads();
      #pragma omp for schedule(static) ordered
      for (int i=0; i<num_threads; i++) {
        #pragma omp ordered
        {
          for (int c=0; c<mTIClusterNum; c++) {
            mTIClustersMember[c].insert(
              mTIClustersMember[c].end(),
              privateTIClustersMember[c].begin(),
              privateTIClustersMember[c].end()
            );
          }
        }
      }
    }

    std::cout << "cluster filled" << std::endl;

    // sort cluster members from the farthest to centroid
    for (auto &cm: mTIClustersMember) {
      std::sort(cm.begin(), cm.end(),
        [this](int i, int j) {
          return this->mCodeToCCDist[i] > this->mCodeToCCDist[j];
        }
      );
    }

    std::cout << "cluster member sorted" << std::endl;

    mEigenVectorsBeforeBalancing = mEigenVectors;
  }
  #endif

  /* Partial balance of variances across subspaces */
  /* Example with 4 subspaces, with 3 dimensions per subspace:
     Variances per subspace per dimension: 
        [0.4, 0.2, 0.1], [0.15, 0.1, 0.06], [0.05, 0.01, 0.01], [0.01, 0.01, 0.009]

     New strategy, to split the first subspace and swap the values to the rest subspaces
        [0.4, 0.06, 0.01], [0.15, 0.1, 0.2], [0.05, 0.01, 0.1], [0.01, 0.01, 0.009] */
  auto checkSubsVarOrdered = [&eigvalueidx, this]() -> bool {
    Eigen::VectorXf eigValPerDim(eigvalueidx.size());
    for (int i=0; i<(int)eigvalueidx.size(); i++) {
      eigValPerDim(i) = eigvalueidx[i].value.real();
    }
    Eigen::VectorXf eigValPerSubs(mSubspaceNum);
    for (int i=0; i<mSubspaceNum; i++) {
      eigValPerSubs(i) = eigValPerDim.segment(
        i*mSubsLen, mSubsLen
      ).sum();
    }

    // is sorted in descending
    return std::is_sorted(eigValPerSubs.data(), eigValPerSubs.data() + mSubspaceNum, 
      [](float lhs, float rhs) -> bool{
        return lhs > rhs;
    });
  };

  int maxsubswap = std::min(mSubsLen, mSubspaceNum);
  for (int i=1; i<maxsubswap; i++) {
    // swap eigenvalue
    EigValueIdxPair tempp = eigvalueidx[i];
    eigvalueidx[i] = eigvalueidx[i*mSubsLen + (mSubsLen-1)];
    eigvalueidx[i*mSubsLen + (mSubsLen-1)] = tempp;
    if (!checkSubsVarOrdered()) {
      // redo swapping
      EigValueIdxPair tempp = eigvalueidx[i];
      eigvalueidx[i] = eigvalueidx[i*mSubsLen + (mSubsLen-1)];
      eigvalueidx[i*mSubsLen + (mSubsLen-1)] = tempp;
      break;
    }
    
    // swap eigenvectors column
    RowVector<Eigen::scomplex> tempc = mEigenVectors.col(i);
    mEigenVectors.col(i) = mEigenVectors.col(i*mSubsLen + (mSubsLen-1));
    mEigenVectors.col(i*mSubsLen + (mSubsLen-1)) = tempc;
  }
  
  #ifndef VAQ_OPTIMIZE
  mEigenVectorsReal = mEigenVectors.real();
  #endif

  // Code for next project
  #if 0
  if (mMethods & NNMethod::TI) {
    metadata = this->ProjectOnFirstDimsEigenVectors(XTrain, mTISegmentNum * mSubsLen);
  }
  #endif

  START_TIMING(PROJECTION);
  this->ProjectOnEigenVectorsInPlace(XTrain, /*withChecking =*/ false);
  END_TIMING_V(PROJECTION, "== PROJECTION computation time: ", verbose);

  /* Allocate bits per dimension */

  /* PercentVarExplained = 0.99 or 0.95 of the variance explained */
  /* Variance explained per PCA dimension (without segmenting anything yet) */
  Eigen::VectorXf varExplainedPerDim(eigvalueidx.size());
  for (int i=0; i<(int)eigvalueidx.size(); i++) {
    varExplainedPerDim(i) = eigvalueidx[i].value.real();
  }
  varExplainedPerDim /= varExplainedPerDim.sum();
  if (verbose)
    std::cout << "Variance: " << varExplainedPerDim.transpose().segment(0, std::min(10, (int)varExplainedPerDim.cols())) << std::endl;
  // get rid of negative eigvalues (if any)
  for (int i=0; i<(int)eigvalueidx.size(); i++) {
    if (varExplainedPerDim(i) < 0.000000000001) {
      varExplainedPerDim(i) = 0.000000000001;
    }
  }

  Eigen::VectorXf varExplainedPerSubs(mSubspaceNum);
  for (int i=0; i<mSubspaceNum; i++) {
    varExplainedPerSubs(i) = varExplainedPerDim.segment(
      i*mSubsLen, mSubsLen
    ).sum();
  }
  std::cout << "VarExplainedPerSubs: " << varExplainedPerSubs.transpose().segment(0, std::min(10, (int)varExplainedPerSubs.cols())) << std::endl;
  mCumSumVarExplainedPerSubs = cumSum(varExplainedPerSubs);
  
  mHighestSubs = 0;
  if (mPercentVarExplained < 1) {
    for (int i=0; i<mSubspaceNum; i++) {
      if (mCumSumVarExplainedPerSubs[i] <= mPercentVarExplained) {
        mHighestSubs = i;
      }
    }
    mHighestSubs += 1;
  } else {
    mHighestSubs = mSubspaceNum;
  }

  mTotalDim = mHighestSubs * mSubsLen;

  if (!verbose) {
    glp_term_out(GLP_OFF);
  }

  /* Solve ILP to assign bits to dimensions
    c_i = variance explained in i dimension
    x_i = (integer) bits allocated in i dimensions
    maximize Sum c_i * x_i */
  glp_prob *lp = glp_create_prob();
  glp_iocp parm; glp_init_iocp(&parm);
  parm.presolve = GLP_ON;
  glp_set_obj_dir(lp, GLP_MAX);
  int glp_rows = mHighestSubs,
      glp_cols = mHighestSubs;
  glp_add_rows(lp, glp_rows);
  glp_add_cols(lp, glp_cols);

  int lastMatIdx = 1, rowCounter = 1;
  std::vector<int> rowIndices(glp_rows * glp_cols + 1, 0), colIndices(glp_rows * glp_cols + 1, 0);
  std::vector<double> numVal(glp_rows * glp_cols + 1, 0);

  // Function
  for (int i=0; i<glp_cols; i++) {
    glp_set_obj_coef(lp, i+1, varExplainedPerSubs(i));
  }

  // set integer constraint
  for (int d=1; d<=glp_cols; d++) {
    glp_set_col_kind(lp, d, GLP_IV);
  }

  /* CONSTRAINTS */
  /* 1. Bit allocation per dimension 0 or more
        x_i >= 0
        x_i <= 8
  */
  for (int i=1; i<=glp_cols; i++) {
    double lb = 0;
    if (mCumSumVarExplainedPerSubs[i-1] <= mPercentVarExplained) { 
      lb = mMinBitsPerSubs;
    }
    glp_set_col_bnds(lp, i, GLP_DB, lb, mMaxBitsPerSubs);
  }

  // sum(x_i) = budget
  // 1 1 1 1 1 = budget
  glp_set_row_bnds(lp, rowCounter, GLP_FX, mBitBudget, 0.0);
  for (int d = 1; d <= glp_cols; d++) {
    rowIndices[lastMatIdx] = rowCounter;
    colIndices[lastMatIdx] = d;
    numVal[lastMatIdx] = 1.0;
    lastMatIdx++;
  }
  rowCounter++;

  /* Force use of at least X% of the variance explained with a uniform single 
    bit allocation scheme of size BitBudget, the most basic scheme
  
    if BitBudget>length(VarExplainedPerDim) 
    UniformAllocVarExplained = CumSumVarExplainedPerDim(BitBudget);
  
    VarExplainedXPercentage = PercentVarExplained*UniformAllocVarExplained;
  
    Force to use dimensions such that VarExplainedXPercentage is satisfied
    for maximization problem we would set x_i >= 1, now we have - x_i <= -1
    HERE WE CONSIDER AT LEAST 2 BITS PER DIMENSION */
  if (verbose) std::cout << "k: ";
  for (int i=0; i<glp_cols-1; i++) {
    int k = nextPow2(varExplainedPerSubs(i) / varExplainedPerSubs(i+1));
    // int k = nextPow2(varExplainedPerSubs(0) / varExplainedPerSubs(i+1));
    if (verbose) std::cout << k << ", ";
    if (std::isnan(k) || k <= 0) {
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
  if (verbose) std::cout << std::endl;

  glp_load_matrix(lp, lastMatIdx-1, rowIndices.data(), colIndices.data(), numVal.data());
  if (verbose) {
    std::cout << "glp matrix: " << std::endl;
    for (int i=0; i<glp_rows; i++) {
      for (int j=0; j<glp_cols; j++)  {
        std::cout << numVal.at(i*glp_cols + j + 1) << ", ";
      }
      int bnds = glp_get_row_type(lp, i+1);
      if (bnds == GLP_FR) {
        std::cout << ": FR";
      } else if (bnds == GLP_LO) {
        std::cout << "> " << glp_get_row_lb(lp, i+1);
      } else if (bnds == GLP_UP) {
        std::cout << "< " << glp_get_row_ub(lp, i+1);
      } else if (bnds == GLP_DB) {
        std::cout << ": DB " << glp_get_row_lb(lp, i+1) << " " << glp_get_row_ub(lp, i+1);
      } else if (bnds == GLP_FX) {
        std::cout << "= " << glp_get_row_lb(lp, i+1);
      }
      std::cout << std::endl;
    }
  }

  int ret = glp_intopt(lp, &parm);
  if (ret != 0) {
    if (verbose) std::cout << "glp solver failed: " << ret << std::endl;
    assert(false);
  }
  if (verbose)
    std::cout << "glp objective value: " << glp_mip_obj_val(lp) << std::endl;


  int totalBit = 0;
  if (mBitsAlloc.size() == 0) {
    mBitsAlloc.resize(glp_cols, 0);
    if (verbose) std::cout << "solution from glp: ";
    for (int i=1; i<=glp_cols; i++) {
      mBitsAlloc[i-1] = glp_mip_col_val(lp, i);
      totalBit += mBitsAlloc[i-1];

      if (verbose) std::cout << mBitsAlloc[i-1] << ",";
    }
    if (verbose) std::cout << std::endl;
    
    // if sum(bit allocation) < bit budget, more likely won't happen
    if (totalBit < mBitBudget) {
      int it = 0;
      while ((it < mHighestSubs) && mBitsAlloc[it] > 0) {
        it++;
      }
      if (it < mHighestSubs) {
        while ((it < mHighestSubs) && mBitsAlloc[it] == 0) {
          mBitsAlloc[it] += 1;
          totalBit++;
          it++;
        }
      }

      if (totalBit < mBitBudget) {
        for (int i=0; i<mHighestSubs; i++) {
          if (mBitsAlloc[i] < mMaxBitsPerSubs) {
            mBitsAlloc[i] += 1;
            totalBit++;
            if (totalBit >= mBitBudget) {
              break;
            }
          }
        }
      }
    }
    if (verbose) std::cout << "Total Bit allocated: " << totalBit << std::endl;
  } else {
    for (auto b: mBitsAlloc) {
      totalBit += b;
    }
    if (verbose) std::cout << "Total Bit allocated: " << totalBit << std::endl;
  }
  if (totalBit < mBitBudget) {
    if (verbose) std::cout << "Error: total bit allocated < bit budget (" << totalBit << " < " << mBitBudget << ")" << std::endl;
    assert(false);
  }

  if (verbose) {
    std::cout << "bit allocation: ";
    for (auto b: mBitsAlloc) {
      std::cout << b << ", ";
    }
    std::cout << std::endl;
  }

  mCentroidsNum.reserve(mBitsAlloc.size());
  for (auto b: mBitsAlloc) {
    mCentroidsNum.push_back((1 << b));
  }

  glp_delete_prob(lp);

  // learn dictionary
  if (verbose) std::cout << "mHighestSubs: " << mHighestSubs << std::endl;
  if (mCentroidsPerSubs.size() == 0) {
    mCentroidsPerSubs.resize(mBitsAlloc.size());
    const int standardBitAlloc = 8;
    for (int iSubs=0; iSubs<mHighestSubs; iSubs++) {
      const int currCentroidsNum = mCentroidsNum[iSubs];
      bool isBitsGtStandard = (mBitsAlloc[iSubs] > standardBitAlloc);

      int sampleSize = std::max(currCentroidsNum * 256, 256*(1 << (mBitBudget/mSubspaceNum)));
      sampleSize = std::min(sampleSize, (int)XTrain.rows());
      RowMatrixXf XTrainSlice(sampleSize, mSubsLen);
      if (sampleSize == XTrain.rows()) {
        std::vector<int> perm(XTrain.rows());
        randomPermutation(perm);
        for (int i=0; i<sampleSize; i++) {
          XTrainSlice.row(i).noalias() = XTrain.block(perm[i], iSubs * mSubsLen, 1, mSubsLen);
        }
      }

      if (mHierarchicalKmeans && isBitsGtStandard) {
        omp_set_num_threads(1);
        if (verbose) std::cout << "subspace " << iSubs << " use hierarchical kmeans" << std::endl;
        int startK = 1 << (standardBitAlloc-1); // start from 128 centroids
        CentroidsMatType currCentroids(startK, mSubsLen);
        arma::fmat means(currCentroids.data(), mSubsLen, startK, false, false);

        arma::fmat data(XTrainSlice.data(), XTrainSlice.cols(), XTrainSlice.rows(), false, false);

        bool status = arma::kmeans(means, data, startK, arma::static_subset, 25, false);
        if (status == false) {
          std::cout << "hierarchical kmeans failed" << std::endl;
          assert(false);
        }

        std::vector<std::vector<int>> members_centroids = getBelongsToCluster(
          XTrainSlice,
          currCentroids
        );

        mCentroidsPerSubs[iSubs].resize(currCentroidsNum, mSubsLen);

        int tempCentroidsRows = (1 << (mBitsAlloc[iSubs] - (standardBitAlloc-1)));
        for (int i=0; i<startK; i++) {
          RowMatrixXf tempXTrain;
          if (members_centroids[i].size() <= 0) {
            std::cout << "hierarchical kmeans failed" << std::endl;
            std::cout << "member centroids " << i << " = 0" << std::endl;
            assert(false);
          }
          tempXTrain.resize(members_centroids[i].size(), mSubsLen);

          for (int rowIdx=0; rowIdx<(int)members_centroids[i].size(); rowIdx++) {
            tempXTrain.row(rowIdx) = XTrainSlice.row(members_centroids[i][rowIdx]);
          }

          CentroidsMatType tempCentroids(tempCentroidsRows, mSubsLen);
          arma::fmat tempMeans(tempCentroids.data(), mSubsLen, tempCentroidsRows, false, false);

          arma::fmat tempData(tempXTrain.data(), tempXTrain.cols(), tempXTrain.rows(), false, false);

          bool status = arma::kmeans(tempMeans, tempData, tempCentroidsRows, arma::static_subset, 25, false);
          if (status == false) {
            std::cout << "hierarchical kmeans failed" << std::endl;
            assert(false);
          }

          mCentroidsPerSubs[iSubs].block(i * tempCentroidsRows, 0, tempCentroids.rows(), mSubsLen) = tempCentroids;
        }
        
        // maybe for the next project        
        #if 0
        // last update
        RowMatrixXf XTrainFullSlice = XTrain.block(0, iSubs * mSubsLen, XTrain.rows(), mSubsLen);
        arma::fmat allmeans(mCentroidsPerSubs[iSubs].data(), mSubsLen, currCentroidsNum, false, false);
        arma::fmat alldata(XTrainFullSlice.data(), XTrainFullSlice.cols(), XTrainFullSlice.rows(), false, false);
        status = arma::kmeans(allmeans, alldata, currCentroidsNum, arma::keep_existing, 1, false);
        if (status == false) {
          std::cout << "last update fail" << std::endl;
          exit(0);
        }
        #endif
      } else if (mBinaryKmeans && isBitsGtStandard) {
        omp_set_num_threads(1);
        if (verbose)
          std::cout << "subspace " << iSubs << " use hierarchical binary kmeans" << std::endl;
        
        mCentroidsPerSubs[iSubs] = hierarchicalBinKmeans(
          XTrainSlice,
          1,
          mBitsAlloc[iSubs]
        );
      } else {
        omp_set_num_threads(1);
        if (verbose) 
          std::cout << "subspace " << iSubs << " regular kmeans" << std::endl;
        mCentroidsPerSubs[iSubs].resize(currCentroidsNum, mSubsLen);
        arma::fmat means(mCentroidsPerSubs[iSubs].data(), mSubsLen, currCentroidsNum, false, false);

        arma::fmat data(XTrainSlice.data(), XTrainSlice.cols(), XTrainSlice.rows(), false, false);

        bool status = arma::kmeans(means, data, currCentroidsNum, arma::static_subset, 25, false);
        if (status == false) {
          std::cout << "kmeans arma failed" << std::endl;
          exit(0);
        }

        // Maybe for the next project
        #if 0
        // last update
        RowMatrixXf XTrainFullSlice = XTrain.block(0, iSubs * mSubsLen, XTrain.rows(), mSubsLen);
        arma::fmat alldata(XTrainFullSlice.data(), XTrainFullSlice.cols(), XTrainFullSlice.rows(), false, false);
        status = arma::kmeans(means, alldata, currCentroidsNum, arma::keep_existing, 1, false);
        if (status == false) {
          std::cout << "last update fail" << std::endl;
          exit(0);
        }

        mCentroidsPerSubs[iSubs] = KMeans::staticFitSampling(
          XTrain.block(0, iSubs * mSubsLen, XTrain.rows(), mSubsLen),
          currCentroidsNum,
          25,
          verbose
        );
        #endif
      }
    }
  }
  // create centroids col major version to enable fast LUT creation
#ifdef __AVX2__
  mCentroidsPerSubsCMajor.resize(mCentroidsPerSubs.size());
  for (int i=0; i<(int)mCentroidsPerSubs.size(); i++) {
    mCentroidsPerSubsCMajor[i] = mCentroidsPerSubs[i];
  }
#endif
}

void VAQ::encode(const RowMatrixXf &XTrain) {
  mXTrainRows = XTrain.rows();
  mXTrainCols = XTrain.cols();
  if (mMethods & NNMethod::Fast) {
    constexpr int batchSize = 32;
    int rowPadding = (int) (std::ceil((float)mXTrainRows / batchSize) * batchSize - mXTrainRows);
    mCodebookCMajor.resize(mXTrainRows + rowPadding, mXTrainCols);
    mCodebookCMajor.setZero();
  } else if (mMethods & NNMethod::Fast2) {
    constexpr int batchSize = 8;
    int rowPadding = (int) (std::ceil((float)mXTrainRows / batchSize) * batchSize - mXTrainRows);
    mCodebookCMajor16.resize(mXTrainRows + rowPadding, mXTrainCols);
    mCodebookCMajor16.setZero();
  } else if (mMethods & NNMethod::Fast3) {
    mStartShufIdx = mBitsAlloc.size();
    for (size_t i=0; i<mBitsAlloc.size(); i++) {
      if (mBitsAlloc[i] > 4) {
        mStartShufIdx = i;
      }
    }
    mStartShufIdx += 1;
    
    // gather batch
    constexpr int gatherBatchSize = 8;
    int gatherRowPadding = (int) (std::ceil((float)mXTrainRows / gatherBatchSize) * gatherBatchSize - mXTrainRows);
    mCodebookCMajor16.resize(mXTrainRows + gatherRowPadding, mStartShufIdx);
    mCodebookCMajor16.setZero();

    // shuffle batch
    constexpr int shufBatchSize = 32;
    int shuffleRowPadding = (int) (std::ceil((float)mXTrainRows / shufBatchSize) * shufBatchSize - mXTrainRows);
    mCodebookCMajor.resize(mXTrainRows + shuffleRowPadding, mXTrainCols - mStartShufIdx);
    mCodebookCMajor.setZero();
  } else if (mMethods & NNMethod::Fast4) {
    mStartShufIdx = mBitsAlloc.size();
    for (size_t i=0; i<mBitsAlloc.size(); i++) {
      if (mBitsAlloc[i] > 4) {
        mStartShufIdx = i;
      }
    }
    mStartShufIdx += 1;

    mCodebook.resize(mXTrainRows, mHighestSubs);

    // shuffle batch
    constexpr int shufBatchSize = 32;
    int shuffleRowPadding = (int) (std::ceil((float)mXTrainRows / shufBatchSize) * shufBatchSize - mXTrainRows);
    mCodebookCMajor.resize(mXTrainRows + shuffleRowPadding, mXTrainCols - mStartShufIdx);
    mCodebookCMajor.setZero();
  } else {
    mCodebook.resize(mXTrainRows, mHighestSubs);
  }

  
  if (mMethods & NNMethod::Fast) {
    encodeImpl(XTrain, mCodebookCMajor);
  } else if (mMethods & NNMethod::Fast2) {
    encodeImpl(XTrain, mCodebookCMajor16);
  } else if (mMethods & NNMethod::Fast3) {
    encodeImplFast3(XTrain);
  } else {
    encodeImpl(XTrain, mCodebook);
  }
}

template<class T>
void VAQ::encodeImpl(const RowMatrixXf &XTrain, T &codebook) {
  // For each subspace
  for (int i=0; i<mHighestSubs; i++) {
    // for each row
    #pragma omp parallel for
    for (int rowIdx=0; rowIdx<mXTrainRows; rowIdx++) {
      uint16_t bestCode = 0;
      float bsf = std::numeric_limits<float>::max();
      for (uint16_t code=0; code<static_cast<uint16_t>(mCentroidsNum[i]); code++) {
        float dist = (XTrain.block(rowIdx, i * mSubsLen, 1, mSubsLen) - mCentroidsPerSubs[i].block(code, 0, 1, mSubsLen)).squaredNorm();

        if (dist < bsf) {
          bestCode = code;
          bsf = dist;
        }
      }
      codebook(rowIdx, i) = bestCode;
    }
  }
}

void VAQ::encodeImplFast3(const RowMatrixXf &XTrain) {
  // For each subspace
  for (int i=0; i<mHighestSubs; i++) {
    // for each row
    #pragma omp parallel for
    for (int rowIdx=0; rowIdx<mXTrainRows; rowIdx++) {
      uint16_t bestCode = 0;
      float bsf = std::numeric_limits<float>::max();
      for (uint16_t code=0; code<static_cast<uint16_t>(mCentroidsNum[i]); code++) {
        float dist = (XTrain.block(rowIdx, i * mSubsLen, 1, mSubsLen) - mCentroidsPerSubs[i].block(code, 0, 1, mSubsLen)).squaredNorm();

        if (dist < bsf) {
          bestCode = code;
          bsf = dist;
        }
      }

      if (i < mStartShufIdx) {
        mCodebookCMajor16(rowIdx, i) = bestCode;
      } else {
        mCodebookCMajor(rowIdx, i-mStartShufIdx) = bestCode;
      }
    }
  }
}

LabelDistVecF VAQ::search(const RowMatrixXf &XTest, const int k, bool verbose) {
  RowMatrixXf XTestPCA = this->ProjectOnEigenVectors(XTest);

  long totalPruned = 0;
  LUTType lut(1 << mMaxBitsPerSubs, mHighestSubs);
  LabelDistVecF ret;
  ret.labels.resize(k * XTest.rows());
  ret.distances.resize(k * XTest.rows());
  f::float_maxheap_t answers = { size_t(XTest.rows()), size_t(k), ret.labels.data(), ret.distances.data() };

  for (int q_idx=0; q_idx < (int)XTestPCA.rows(); q_idx++) {
    switch (mMaxBitsPerSubs) {
      case 9: CreateLUT<9>(XTestPCA.row(q_idx), lut); break;
      case 10: CreateLUT<10>(XTestPCA.row(q_idx), lut); break;
      case 11: CreateLUT<11>(XTestPCA.row(q_idx), lut); break;
      case 12: CreateLUT<12>(XTestPCA.row(q_idx), lut); break;
      case 13: CreateLUT<13>(XTestPCA.row(q_idx), lut); break;
      case 14: CreateLUT<14>(XTestPCA.row(q_idx), lut); break;
      case 15: CreateLUT<15>(XTestPCA.row(q_idx), lut); break;
      default:
        CreateLUT(XTestPCA.row(q_idx), lut);
        break;
    }
    if (mMethods & NNMethod::TI) {
      std::vector<float> qToCCDist(mTIClusters.rows());
      const float* qptr = XTestPCA.row(q_idx).data();
      fvec_L2sqr_ny(qToCCDist.data(), qptr, mTIClusters.data(), mTISegmentNum * mSubsLen, mTIClusters.rows());
      for (int i=0; i<(int)qToCCDist.size(); i++) {
        qToCCDist[i] = std::sqrt(qToCCDist[i]);
      }

      // Maybe for the next project
      #if 0
      RowVectorXf XTestPCAUnbalanced = (XTest.row(q_idx) * mEigenVectorsBeforeBalancing.block(0, 0, mEigenVectorsBeforeBalancing.rows(), mTISegmentNum * mSubsLen)).real();
      for (int i=0; i<(int)qToCCDist.size(); i++) {
        qToCCDist[i] = (XTestPCAUnbalanced - mTIClusters.row(i)).norm();
      }
      #endif

      Eigen::VectorXi qToCCIdx = Eigen::VectorXi::LinSpaced(mTIClusters.rows(), 0, mTIClusters.rows()-1);
      std::sort(qToCCIdx.data(), qToCCIdx.data()+qToCCIdx.size(), 
        [&qToCCDist](int i, int j) -> bool {
          return qToCCDist[i] < qToCCDist[j];
        }
      );
      long prunedPerQuery = 0;
      // Maybe for the next project
      #if 0
      = searchTriangleInequality(XTestPCAUnbalanced, lut, k, mMethods, qToCCIdx, qToCCDist, prunedPerQuery);
      #endif
      searchTriangleInequality(lut, k, mMethods, qToCCIdx, qToCCDist, prunedPerQuery, q_idx, answers);
      totalPruned += prunedPerQuery;
    } else if (mMethods & NNMethod::EA) {
      searchEarlyAbandon(lut, k, q_idx, answers);
    } else if (mMethods & NNMethod::Heap) {
      searchHeap(lut, k, q_idx, answers);
    } else if (mMethods & NNMethod::Fast) {
      vectorIdxDistPairConverterFloatHeap<int16_t>(searchFast(lut, k), answers, q_idx * k);
    } else if (mMethods & NNMethod::Fast2) {
      searchFast2(lut, k, q_idx, answers);
    } else if (mMethods & NNMethod::Fast3) {
      searchFast3(lut, k, q_idx, answers);
    } else {
      searchSort(lut, k, q_idx, answers);
    }
  }
  if (verbose && (mMethods & NNMethod::TI)) {
    std::cout << "Average pruned per query: " << ((float)totalPruned / XTestPCA.rows()) << std::endl;
  }

  return ret;
}

LabelDistVecF VAQ::refine(const RowMatrixXf &XTest, const LabelDistVecF &answersIn, const RowMatrixXf &XTrain, const int k) {
  int refineNum = answersIn.labels.size() / XTest.rows();
  LabelDistVecF ret;
  ret.labels.resize(XTest.rows() * k);
  ret.distances.resize(XTest.rows() * k);
  f::float_maxheap_t answers = {
    size_t(XTest.rows()), size_t(k), ret.labels.data(), ret.distances.data()
  };
  
  // heapmax
  for (int q_idx=0; q_idx < (int)XTest.rows(); q_idx++) {
    int * __restrict heap_ids = answers.ids + q_idx * k;
    float * __restrict heap_dis = answers.val + q_idx * k;

    f::heap_heapify<f::CMax<float, int>> (k, heap_dis, heap_ids);

    for (int i=0; i<refineNum; i++) {
      float dist = (XTest.row(q_idx) - XTrain.row(answersIn.labels[q_idx * refineNum + i])).squaredNorm();
      if (f::CMax<float, int>::cmp(heap_dis[0], dist)) {
        f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
        f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, answersIn.labels[q_idx * refineNum + i]);
      }
    }
    f::heap_reorder<f::CMax<float, int>> (k, heap_dis, heap_ids);
  }

  return ret;
}

void VAQ::clusterTI(bool useKMeans, bool verbose) {
  if (mTIVariance < 1) {
    mTISegmentNum = 0;
    for (int i=0; i<mSubspaceNum; i++) {
      if (mCumSumVarExplainedPerSubs[i] <= mTIVariance)  {
        mTISegmentNum += 1;
      }
    }
    if (mTISegmentNum == 0) {
      mTISegmentNum = 1;
    }
    std::cout << "from " << mTIVariance << " variance, get " << mTISegmentNum << " segment" << std::endl;
  } else if (mTISegmentNum == -1) {
    mTISegmentNum = mHighestSubs;
  }
  mTIClusters.resize(mTIClusterNum, mTISegmentNum * mSubsLen);
  mTIClusters.setZero();

  if (useKMeans) {
    mTIClusters = KMeans::staticFitCodebook(
      mCodebook.block(0, 0, mCodebook.rows(), mTISegmentNum), 
      mCentroidsPerSubs, mTISegmentNum, mTIClusterNum, 50, verbose
    );
  } else {
    for (int i=0; i < mTIClusterNum; i++) {
      int randIdx = rand() % mCodebook.rows();
      for (int subs=0; subs < mTISegmentNum; subs++) {
        mTIClusters.row(i).segment(
          subs*mSubsLen, mSubsLen
        ) = mCentroidsPerSubs[subs].row(mCodebook(randIdx, subs));
      }
    }
  }

  std::cout << "cluster created" << std::endl;

  mCodeToCCDist.resize(mCodebook.rows());
  mTIClustersMember.resize(mTIClusterNum);
  for (int i=0; i<(int)mTIClusters.rows(); i++) {
    mTIClustersMember[i].reserve(mCodebook.rows() / mTIClusterNum);
  }
  START_TIMING(FILLING_CLUSTER);
  constexpr int threadnum = 1;
  #pragma omp parallel num_threads(threadnum)
  {
    std::vector<std::vector<int>> privateTIClustersMember(mTIClusterNum);
    for (int i=0; i<mTIClusterNum; i++) {
      privateTIClustersMember[i].reserve(mCodebook.rows() / (mTIClusterNum*threadnum));
    }

    #pragma omp for nowait schedule(static)
    for (int i=0; i<mCodebook.rows(); i++) {
      RowVectorXf x(mTISegmentNum * mSubsLen);
      for (int subs=0; subs < mTISegmentNum; subs++) {
        x.segment(
          subs*mSubsLen, mSubsLen
        ) = mCentroidsPerSubs.at(subs).row(mCodebook(i, subs));
      }

      float closestDist = std::numeric_limits<float>::max();
      int closestIdx = -1;
      std::vector<float> dists(mTIClusterNum);
      fvec_L2sqr_ny(dists.data(), x.data(), mTIClusters.data(), x.size(), mTIClusterNum);
      for (int c=0; c<mTIClusterNum; c++) {
        // float dist = (x - mTIClusters.row(c)).norm();
        float dist = std::sqrt(dists[c]);
        if (dist < closestDist) {
          closestIdx = c;
          closestDist = dist;
        }
      }
      mCodeToCCDist.at(i) = closestDist;
      privateTIClustersMember[closestIdx].push_back(i);
    }

    const int num_threads = omp_get_num_threads();
    #pragma omp for schedule(static) ordered
    for (int i=0; i<num_threads; i++) {
      #pragma omp ordered
      {
        for (int c=0; c<mTIClusterNum; c++) {
          mTIClustersMember[c].insert(
            mTIClustersMember[c].end(),
            privateTIClustersMember[c].begin(),
            privateTIClustersMember[c].end()
          );
        }
      }
    }
  }
  END_TIMING(FILLING_CLUSTER, "Filling cluster = ");

  std::cout << "cluster filled" << std::endl;

  // sort cluster members from the farthest to centroid
  for (auto &cm: mTIClustersMember) {
    std::sort(cm.begin(), cm.end(),
      [this](int i, int j) {
        return this->mCodeToCCDist[i] > this->mCodeToCCDist[j];
      }
    );
  }

  std::cout << "cluster member sorted" << std::endl;

  // sort group the codebook
  mClusterMembersStartIdx.resize(mTIClustersMember.size());
  CodebookType groupedCodebook(mCodebook.rows(), mCodebook.cols());
  int clusterMemberIdx = 0;
  int rowCounter = 0;
  for (auto &cm: mTIClustersMember) {
    mClusterMembersStartIdx[clusterMemberIdx] = rowCounter;
    for (const int idx: cm) {
      groupedCodebook.row(rowCounter) = mCodebook.row(idx);
      rowCounter++;
    }
    clusterMemberIdx++;
  }
  mCodebook = groupedCodebook;

  std::cout << "codebook sort grouped" << std::endl;
}

// Maybe for next project
#if 0
void VAQ::clusterTI(bool useKMeans, bool verbose) {
  // if (mTIVariance < 1) {
  //   mTISegmentNum = 0;
  //   for (int i=0; i<mSubspaceNum; i++) {
  //     if (mCumSumVarExplainedPerSubs[i] <= mTIVariance)  {
  //       mTISegmentNum += 1;
  //     }
  //   }
  // } else if (mTISegmentNum == -1) {
  //   mTISegmentNum = mHighestSubs;
  // }
  // mTIClusters.resize(mTIClusterNum, mTISegmentNum * mSubsLen);
  // mTIClusters.setZero();

  // if (useKMeans) {
  //   mTIClusters = KMeans::staticFitCodebook(
  //     mCodebook.block(0, 0, mCodebook.rows(), mTISegmentNum), 
  //     mCentroidsPerSubs, mTISegmentNum, mTIClusterNum, 50, verbose
  //   );
  // } else {
  //   for (int i=0; i < mTIClusterNum; i++) {
  //     int randIdx = rand() % mCodebook.rows();
  //     for (int subs=0; subs < mTISegmentNum; subs++) {
  //       mTIClusters.row(i).segment(
  //         subs*mSubsLen, mSubsLen
  //       ) = mCentroidsPerSubs[subs].row(mCodebook(randIdx, subs));
  //     }
  //   }
  // }

  // std::cout << "cluster created" << std::endl;

  // mCodeToCCDist.resize(mCodebook.rows());
  // mTIClustersMember.resize(mTIClusterNum);
  // for (int i=0; i<(int)mTIClusters.rows(); i++) {
  //   mTIClustersMember[i].reserve(mCodebook.rows() / mTIClusterNum);
  // }
  // constexpr int threadnum = 1;
  // #pragma omp parallel num_threads(threadnum)
  // {
  //   std::vector<std::vector<int>> privateTIClustersMember(mTIClusterNum);
  //   for (int i=0; i<mTIClusterNum; i++) {
  //     privateTIClustersMember[i].reserve(mCodebook.rows() / (mTIClusterNum*threadnum));
  //   }

  //   #pragma omp for nowait schedule(static)
  //   for (int i=0; i<mCodebook.rows(); i++) {
  //     RowVectorXf x(mTISegmentNum * mSubsLen);
  //     for (int subs=0; subs < mTISegmentNum; subs++) {
  //       x.segment(
  //         subs*mSubsLen, mSubsLen
  //       ) = mCentroidsPerSubs.at(subs).row(mCodebook(i, subs));
  //     }

  //     float closestDist = std::numeric_limits<float>::max();
  //     int closestIdx = -1;
  //     for (int c=0; c<mTIClusterNum; c++) {
  //       float dist = (x - mTIClusters.row(c)).norm();
  //       if (dist < closestDist) {
  //         closestIdx = c;
  //         closestDist = dist;
  //       }
  //     }
  //     mCodeToCCDist.at(i) = closestDist;
  //     privateTIClustersMember[closestIdx].push_back(i);
  //   }

  //   const int num_threads = omp_get_num_threads();
  //   #pragma omp for schedule(static) ordered
  //   for (int i=0; i<num_threads; i++) {
  //     #pragma omp ordered
  //     {
  //       for (int c=0; c<mTIClusterNum; c++) {
  //         mTIClustersMember[c].insert(
  //           mTIClustersMember[c].end(),
  //           privateTIClustersMember[c].begin(),
  //           privateTIClustersMember[c].end()
  //         );
  //       }
  //     }
  //   }
  // }

  // std::cout << "cluster filled" << std::endl;

  // // sort cluster members from the farthest to centroid
  // for (auto &cm: mTIClustersMember) {
  //   std::sort(cm.begin(), cm.end(),
  //     [this](int i, int j) {
  //       return this->mCodeToCCDist[i] > this->mCodeToCCDist[j];
  //     }
  //   );
  // }

  // std::cout << "cluster member sorted" << std::endl;

  // sort group the codebook
  mClusterMembersStartIdx.resize(mTIClustersMember.size());
  CodebookType groupedCodebook(mCodebook.rows(), mCodebook.cols());
  int clusterMemberIdx = 0;
  int rowCounter = 0;
  for (auto &cm: mTIClustersMember) {
    mClusterMembersStartIdx[clusterMemberIdx] = rowCounter;
    for (const int idx: cm) {
      groupedCodebook.row(rowCounter) = mCodebook.row(idx);
      rowCounter++;
    }
    clusterMemberIdx++;
  }
  mCodebook = groupedCodebook;

  std::cout << "codebook sort grouped" << std::endl;
}
#endif

void VAQ::learnQuantization(const RowMatrixXf &XTrain, float sampleRatio) {
  RowMatrixXf XTrainPCA = this->ProjectOnEigenVectors(XTrain);
  const int sampleSize = static_cast<int>(sampleRatio * (float)XTrainPCA.rows());

  int totalDim = mTotalDim;
  if (mMethods & NNMethod::Fast3) {
    for (int i=0; i<mStartShufIdx; i++) {
      totalDim -= mSubsLen;
    }
  }
  if (totalDim == 0) {
    return;
  }
  RowMatrixXf queryLearn(sampleSize, totalDim);

  std::vector<int> perm(XTrainPCA.rows());
  randomPermutation(perm);
  for (int i=0; i<sampleSize; i++) {
    queryLearn.row(i) = XTrainPCA.block(perm[i], mTotalDim-totalDim, 1, totalDim);
  }
  int nShuffleNum = mHighestSubs - mStartShufIdx;

  int lutRows = (int)std::pow(2, mMaxBitsPerSubs);
  LUTType luts(sampleSize * lutRows, nShuffleNum);
  for (int i=0; i<sampleSize; i++) {
    LUTType temp(lutRows, nShuffleNum);
    if (mMethods & NNMethod::Fast3) {
      CreateLUTFast3(queryLearn.row(i), temp, nShuffleNum, mTotalDim-totalDim);
    } else {
      switch (mMaxBitsPerSubs) {
        case 9: CreateLUT<9>(queryLearn.row(i), temp); break;
        case 10: CreateLUT<10>(queryLearn.row(i), temp); break;
        case 11: CreateLUT<11>(queryLearn.row(i), temp); break;
        case 12: CreateLUT<12>(queryLearn.row(i), temp); break;
        case 13: CreateLUT<13>(queryLearn.row(i), temp); break;
        case 14: CreateLUT<14>(queryLearn.row(i), temp); break;
        case 15: CreateLUT<15>(queryLearn.row(i), temp); break;
        default:
          CreateLUT(queryLearn.row(i), temp);
          break;
      }
    }
    luts.block(i * lutRows, 0, lutRows, nShuffleNum) = temp;
  }

  float best_loss = std::numeric_limits<float>::max();
  for (const float& alpha: {.001f, .002f, .005f, .01f, .02f, .05f, .1f}) {
    RowVector<float> floors = percentile(luts, alpha);
    LUTType lut_offset =  (luts.rowwise() - floors).array().max(0);
    ColVector<float> ceil = percentile(lut_offset, 1.0f - alpha);

    ColVector<float> scaleBy = ceil.unaryExpr([](float x){return 255.0f/x;});
    SmallLUTType luts_quantized = smallQuantize(lut_offset, scaleBy);

    // compute error
    LUTType luts_ideal(luts.rows(), luts.cols());
    luts_ideal = (luts.array() - lut_offset.array()).matrix();
    for (int i=0; i<luts_ideal.cols(); i++) {
      luts_ideal.col(i) = (luts_ideal.col(i) * scaleBy(i));
    }
    luts_ideal = (luts_ideal.array() - luts_quantized.cast<float>().array()).matrix();

    float loss = luts_ideal.unaryExpr([](float x){return x*x;}).sum();
    if (loss <= best_loss) {
      best_loss = loss;
      mOffsets = floors;
      mScale = scaleBy;
    }
  }
}

void VAQ::parseMethodString(std::string methodString) {
  std::vector<std::string> parsed;
  std::stringstream ss(methodString);
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    parsed.push_back(substr);
  }

  for (std::string token: parsed) {
    if (token.rfind("VAQ", 0) == 0) {
      int totalBit;
      int subvector;
      int minBits;
      int maxBits;
      float variance;
      if (std::sscanf(token.c_str(), "VAQ%dm%dmin%dmax%dvar%f", &totalBit, &subvector, &minBits, &maxBits, &variance) == 5) {
        mBitBudget = totalBit;
        mSubspaceNum = subvector;
        mMinBitsPerSubs = minBits;
        mMaxBitsPerSubs = maxBits;
        mPercentVarExplained = variance;
      }
    } else if (
      (token.rfind("SORT") != std::string::npos) ||
      (token.rfind("HEAP") != std::string::npos) ||
      (token.rfind("EA") != std::string::npos) ||
      (token.rfind("TI") != std::string::npos) ||
      (token.rfind("FAST") != std::string::npos) ||
      (token.rfind("FAST2") != std::string::npos)
    ) {
      std::vector<std::string> parsedMethod;
      std::stringstream ssmethod(token);
      while (ssmethod.good()) {
        std::string substr;
        std::getline(ssmethod, substr, '_');
        parsedMethod.push_back(substr);
      }

      mMethods = 0;
      for (std::string tokenMethod: parsedMethod) {
        if (tokenMethod.rfind("SORT") != std::string::npos) {
          mMethods |= NNMethod::Sort;
        } else if (tokenMethod.rfind("HEAP") != std::string::npos) {
          mMethods |= NNMethod::Heap;
        } else if (tokenMethod.rfind("EA") != std::string::npos) {
          mMethods |= NNMethod::EA;
        } else if (tokenMethod.rfind("TI") != std::string::npos) {
          size_t cluster, segment;
          float minvar;
          if (std::sscanf(tokenMethod.c_str(), "TI%luvar%f", &cluster, &minvar) == 2) {
            mMethods |= NNMethod::TI;
            mTIClusterNum = cluster;
            mTIVariance = minvar;
          } else if (std::sscanf(tokenMethod.c_str(), "TI%lum%lu", &cluster, &segment) == 2) {
            mMethods |= NNMethod::TI;
            mTIClusterNum = cluster;
            mTISegmentNum = segment;
          } else if (std::sscanf(tokenMethod.c_str(), "TI%lu", &cluster) == 1) {
            mMethods |= NNMethod::TI;
            mTIClusterNum = cluster;
          }
        } else if (tokenMethod.rfind("FAST3") != std::string::npos) {
          mMethods |= NNMethod::Fast3;
        } else if (tokenMethod.rfind("FAST2") != std::string::npos) {
          mMethods |= NNMethod::Fast2;
        } else if (tokenMethod.rfind("FAST") != std::string::npos) {
          mMethods |= NNMethod::Fast;
        }
      }
    }
  }

  // Validation for FAST method
  if ((mMethods & NNMethod::Fast) && (mMaxBitsPerSubs > 4)) {
    std::cout << "Error: max bit per subs couldn't be > 4 when using FAST query method" << std::endl;
    exit(0);
  }
}

/**
 * train Auxiliary Functions
 */
std::vector<std::vector<int>> VAQ::getBelongsToCluster(const RowMatrixXf &X, const CentroidsMatType &C) {
  std::vector<std::vector<int>> members(C.rows());
  for (int i=0; i<C.rows(); i++) {
    members[i].reserve(X.rows() / C.rows());
  }

  for (int rowIdx=0; rowIdx<X.rows(); rowIdx++) {
    int min_idx = -1;
    float min_dist = std::numeric_limits<float>::max();
    for (int centIdx=0; centIdx<C.rows(); centIdx++) {
      float dist = (X.row(rowIdx) - C.row(centIdx)).squaredNorm();
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = centIdx;
      }
    }
    members[min_idx].push_back(rowIdx);
  }

  return members;
}
std::vector<int> VAQ::getBelongsToBinaryCluster(const RowMatrixXf &X, const CentroidsMatType &C, size_t &sizeleft, size_t &sizeright) {
  std::vector<int> members(X.rows());
  sizeleft = 0; sizeright = 0;

  for (int rowIdx=0; rowIdx<X.rows(); rowIdx++) {
    float leftdist = (X.row(rowIdx) - C.row(0)).squaredNorm();
    float rightdist = (X.row(rowIdx) - C.row(1)).squaredNorm();
    if (leftdist < rightdist) {
      members[rowIdx] = 0;
      sizeleft += 1;
    } else {
      members[rowIdx] = 1;
      sizeright += 1;
    }
  }

  return members;
}
CentroidsMatType VAQ::hierarchicalBinKmeans(RowMatrixXf& X, int depth, const int maxdepth) {
  CentroidsMatType currCentroids(2, X.cols());
  arma::fmat means(currCentroids.data(), X.cols(), 2, false, false);
  arma::fmat data(X.data(), X.cols(), X.rows(), false, false);
  bool status = arma::kmeans(means, data, 2, arma::static_subset, 25, false);
  if (status == false) {
    std::cout << "kmeans arma failed, depth = " << depth << std::endl;
    exit(0);
  }

  if (depth == maxdepth) {
    return currCentroids;
  } else {
    size_t sizeleft, sizeright;
    std::vector<int> members_centroids = getBelongsToBinaryCluster(
      X, currCentroids, sizeleft, sizeright
    );
    
    const size_t tempK = 1 << (maxdepth - (depth-1));
    if (sizeleft < tempK/2 || sizeright < tempK/2) {
      // cut the recursive calls
      CentroidsMatType cutCentroids(tempK, X.cols());
      arma::fmat cutmeans(cutCentroids.data(), X.cols(), tempK, false, false);
      bool status = arma::kmeans(cutmeans, data, tempK, arma::static_subset, 25, false);
      if (status == false) {
        std::cout << "kmeans arma failed when cut recursive, depth = " << depth << std::endl;
        exit(0);
      }
      return cutCentroids;
    }

    RowMatrixXf tempXTrainLeft(sizeleft, X.cols());
    RowMatrixXf tempXTrainRight(sizeright, X.cols());
    int ctLeft = 0, ctRight = 0;
    for (int i=0; i<(int)members_centroids.size(); i++) {
      if (members_centroids[i] == 0) {
        tempXTrainLeft.row(ctLeft++) = X.row(i);
      } else {
        tempXTrainRight.row(ctRight++) = X.row(i);
      }
    }


    if (sizeleft < tempK/2) {
      std::cout << "sizeleft = " << sizeleft << ", depth " << depth << std::endl;
    } 
    if (sizeright < tempK/2) {
      std::cout << "sizeright = " << sizeright << ", depth " << depth << std::endl;
    } 

    CentroidsMatType retCentroids(tempK, X.cols());
    retCentroids.block(0, 0, tempK / 2, X.cols()) = hierarchicalBinKmeans(
      tempXTrainLeft, depth+1, maxdepth
    );
    retCentroids.block(tempK / 2, 0, tempK / 2, X.cols()) = hierarchicalBinKmeans(
      tempXTrainRight, depth+1, maxdepth
    );
    return retCentroids;
  }

}

void VAQ::CreateLUTFast3(const RowVectorXf &query, LUTType &lut, const int nShuffleNum, const int nGatherDim) {
  lut.setZero();

  static constexpr int packet_width = 8; // objs per simd register
  for (int subs=0; subs<nShuffleNum; subs++) {
    if (mCentroidsNum[mStartShufIdx+subs] >= 8) { // only vectorized when centroids >= 8
      const int nstripes = (int)(std::ceil(mCentroidsNum[mStartShufIdx+subs] / packet_width));
      __m256 accumulators[8192/8];  // max centroids
      
      auto lut_ptr = lut.data() + lut.rows()*subs;

      for (int i=0; i<nstripes; i++) {
        accumulators[i] = _mm256_setzero_ps();
      }

      for (int j=0; j<mSubsLen; j++) {
        auto centroids_ptr = (mCentroidsPerSubsCMajor[mStartShufIdx+subs]).data() + mCentroidsPerSubsCMajor[mStartShufIdx+subs].rows()*j;
        
        auto q_broadcast = _mm256_set1_ps(query((mStartShufIdx+subs) * mSubsLen + j-nGatherDim));
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
      for (int cIdx=0; cIdx < mCentroidsNum[mStartShufIdx+subs]; cIdx++) {
        lut(cIdx, subs) += (query.segment((mStartShufIdx+subs)*mSubsLen-nGatherDim, mSubsLen) - mCentroidsPerSubs[mStartShufIdx+subs].block(cIdx, 0, 1, mSubsLen)).squaredNorm();
      }
    }
  }
}

// maybe for next project
#if 0
void VAQ::searchTriangleInequality(const RowVectorXf& queryunbalanced, LUTType &lut, const int k, uint32_t methods, const Eigen::VectorXi &qToCCIdx, const std::vector<float> &qToCCDist, long &prunedPerQuery) {
  auto comparator = [](IdxDistPairFloat const& a, IdxDistPairFloat const& b) -> bool {
    return a.dist < b.dist;
  };
  
  std::vector<IdxDistPairFloat> pairs;
  pairs.reserve(k+1);
  std::make_heap(pairs.begin(), pairs.end(), comparator);
  float bsfK = 0;
  float bsfKOri = 0;
  int counter = 0;
  if (methods & NNMethod::EA) { // early abandon
    float bsfKSquared = 0;
    const int maxClusterVisit = qToCCIdx.size() / 1;
    for (int ccIdxIdx = 0; ccIdxIdx < maxClusterVisit; ccIdxIdx++) {
      const int clusterIdx = qToCCIdx[ccIdxIdx];
      const int clusterDataStartIdx = mClusterMembersStartIdx[clusterIdx];
      if (mTIClustersMember[clusterIdx].size() == 0) {
        continue;
      }
      
      auto codes = mCodebook.data() + (mCodebook.cols() * clusterDataStartIdx);
      int interCounter = 0;
      for (const int dataIndex: mTIClustersMember[clusterIdx]) {
        if (counter < k) {
          float dist = 0;
          for (int col=0; col < mHighestSubs; col++) {
            float* luts = lut.data() + lut.rows() * col;
            dist += luts[codes[col]];
          }
          dist = std::sqrt(dist);
          pairs.emplace_back(dataIndex, dist);
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          if (dist > bsfK) {
            bsfK = dist;
            bsfKSquared = bsfK*bsfK;
          }
          counter++;
          if (counter >= k) {
            bsfKOri = (metadata.row(dataIndex) - queryunbalanced).norm();
          }
        } else {
          if (bsfKOri <= (qToCCDist[clusterIdx] - mCodeToCCDist[dataIndex])) {
            prunedPerQuery += mTIClustersMember[clusterIdx].size() - interCounter;
            break;
          }
          float dist = 0;
          for (int col=0; col < mHighestSubs && dist < bsfKSquared; col++) {
            float* luts = lut.data() + lut.rows() * col;
            dist += luts[codes[col]];
          }
          if (dist < bsfKSquared) {
            dist = std::sqrt(dist);
            pairs.emplace_back(dataIndex, dist);
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            std::pop_heap(pairs.begin(), pairs.end(), comparator);
            pairs.pop_back();
            bsfK = (pairs.front()).dist;
            bsfKSquared = bsfK*bsfK;
            bsfKOri = (metadata.row(dataIndex) - queryunbalanced).norm();
          }
        }
        codes += mCodebook.cols();
        interCounter++;
      }
    }
    for (int i=maxClusterVisit; i<qToCCIdx.size(); i++) {
      prunedPerQuery += mTIClustersMember[qToCCIdx[i]].size();
    }
  } else {
    for (int ccIdxIdx = 0; ccIdxIdx < qToCCIdx.size(); ccIdxIdx++) {
      const int clusterIdx = qToCCIdx[ccIdxIdx];
      const int clusterDataStartIdx = mClusterMembersStartIdx[clusterIdx];
      if (mTIClustersMember[clusterIdx].size() == 0) {
        continue;
      }
      auto codes = mCodebook.data() + (mCodebook.cols() * clusterDataStartIdx);
      int interCounter = 0;
      for (const int dataIndex: mTIClustersMember[clusterIdx]) {
        if (counter < k) {
          float dist = 0;
          for (int col=0; col < mHighestSubs; col++) {
            float* luts = lut.data() + lut.rows() * col;
            dist += luts[codes[col]];
          }
          dist = std::sqrt(dist);
          pairs.emplace_back(dataIndex, dist);
          std::push_heap(pairs.begin(), pairs.end(), comparator);
          if (dist > bsfK) {
            bsfK = dist;
          }
          counter++;
          if (counter >= k) {
            bsfKOri = (metadata.row(dataIndex) - queryunbalanced).norm();
          }
        } else {
          if (bsfKOri <= (qToCCDist[clusterIdx] - mCodeToCCDist[dataIndex])) {
            prunedPerQuery += mTIClustersMember[clusterIdx].size() - interCounter;
            break;
          }
          float dist = 0;
          for (int col=0; col < mHighestSubs; col++) {
            float* luts = lut.data() + lut.rows() * col;
            dist += luts[codes[col]];
          }
          dist = std::sqrt(dist);
          if (dist < bsfK) {
            pairs.emplace_back(dataIndex, dist);
            std::push_heap(pairs.begin(), pairs.end(), comparator);
            std::pop_heap(pairs.begin(), pairs.end(), comparator);
            pairs.pop_back();
            bsfK = (pairs.front()).dist;
            bsfKOri = (metadata.row(dataIndex) - queryunbalanced).norm();
          }
        }
        codes += mCodebook.cols();
        interCounter++;
      }
    }
  }
  std::sort_heap(pairs.begin(), pairs.end(), comparator);
  return pairs;
}
#endif
void VAQ::searchTriangleInequality(LUTType &lut, const int k, uint32_t methods, const Eigen::VectorXi &qToCCIdx, const std::vector<float> &qToCCDist, long &prunedPerQuery, int q_idx, f::float_maxheap_t &res) {
  int * __restrict heap_ids = res.ids + q_idx * k;
  float * __restrict heap_dis = res.val + q_idx * k;
  
  f::heap_heapify<f::CMax<float, int>> (k, heap_dis, heap_ids);
  float bsfK = 0;
  int counter = 0;
  
  int maxClusterVisit = qToCCIdx.size();
  if (mVisit < 1) {
    maxClusterVisit = static_cast<int>(static_cast<float>(qToCCIdx.size()) * mVisit);
  }
  bool retrievedEnough = false;
  if (methods & NNMethod::EA) { // early abandon
    float bsfKSquared = 0;
    for (int ccIdxIdx = 0; (ccIdxIdx < maxClusterVisit) || (!retrievedEnough && ccIdxIdx < qToCCIdx.size()); ccIdxIdx++) {
      const int clusterIdx = qToCCIdx[ccIdxIdx];
      const int clusterDataStartIdx = mClusterMembersStartIdx[clusterIdx];
      if (mTIClustersMember[clusterIdx].size() == 0) {
        continue;
      }
      
      auto codes = mCodebook.data() + (mCodebook.cols() * clusterDataStartIdx);
      int interCounter = 0;
      for (const int dataIndex: mTIClustersMember[clusterIdx]) {
        if (counter >= k) {
          if (bsfK <= (qToCCDist[clusterIdx] - mCodeToCCDist[dataIndex])) {
            prunedPerQuery += mTIClustersMember[clusterIdx].size() - interCounter;
            break;
          }
          float dist = 0;
          const float * luts = lut.data();
          const int ksub = lut.rows();
          int col;
          for (col=0; col < mHighestSubs && dist < bsfKSquared; col += 4) {
            float dism = 0;
            dism  = luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dist += dism;
          }
          codes += (mHighestSubs-col);

          if (dist < bsfKSquared) {
            dist = std::sqrt(dist);
            f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
            f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, dataIndex);
            bsfK = heap_dis[0];
            bsfKSquared = bsfK*bsfK;
          }
        } else {
          float dist = 0;
          const float * luts = lut.data();
          const int ksub = lut.rows();
          int col;
          for (col=0; col < mHighestSubs; col += 4) {
            float dism = 0;
            dism  = luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dist += dism;
          }
          dist = std::sqrt(dist);

          f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
          f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, dataIndex);
          if (dist > bsfK) {
            bsfK = dist;
            bsfKSquared = bsfK*bsfK;
          }
          counter++;
        }
        interCounter++;
      }
      if (counter >= k) {
        retrievedEnough = true;
      }
    }
    for (int i=maxClusterVisit; i<qToCCIdx.size(); i++) {
      prunedPerQuery += mTIClustersMember[qToCCIdx[i]].size();
    }
  } else {
    float bsfKSquared = 0;
    for (int ccIdxIdx = 0; (ccIdxIdx < maxClusterVisit) || (!retrievedEnough && ccIdxIdx < qToCCIdx.size()); ccIdxIdx++) {
      const int clusterIdx = qToCCIdx[ccIdxIdx];
      const int clusterDataStartIdx = mClusterMembersStartIdx[clusterIdx];
      if (mTIClustersMember[clusterIdx].size() == 0) {
        continue;
      }
      
      auto codes = mCodebook.data() + (mCodebook.cols() * clusterDataStartIdx);
      int interCounter = 0;
      for (const int dataIndex: mTIClustersMember[clusterIdx]) {
        if (counter >= k) {
          if (bsfK <= (qToCCDist[clusterIdx] - mCodeToCCDist[dataIndex])) {
            prunedPerQuery += mTIClustersMember[clusterIdx].size() - interCounter;
            break;
          }
          const float * luts = lut.data();
          const int ksub = lut.rows();
          float dist = 0;
          for (int col=0; col < mHighestSubs; col += 4) {
            float dism = 0;
            dism  = luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dist += dism;
          }

          if (dist < bsfKSquared) {
            dist = std::sqrt(dist);
            f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
            f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, dataIndex);
            bsfK = heap_dis[0];
            bsfKSquared = bsfK*bsfK;
          }
        } else {
          const float * luts = lut.data();
          const int ksub = lut.rows();
          float dist = 0;
          for (int col=0; col < mHighestSubs; col += 4) {
            float dism = 0;
            dism  = luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dism += luts[*codes++]; luts += ksub;
            dist += dism;
          }
          dist = std::sqrt(dist);

          f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
          f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, dataIndex);
          if (dist > bsfK) {
            bsfK = dist;
          }
          counter++;
        }
        interCounter++;
      }
      if (counter >= k) {
        retrievedEnough = true;
      }
    }
    for (int i=maxClusterVisit; i<qToCCIdx.size(); i++) {
      prunedPerQuery += mTIClustersMember[qToCCIdx[i]].size();
    }
  }
  
  f::heap_reorder<f::CMax<float, int>> (k, heap_dis, heap_ids);
}

void VAQ::searchEarlyAbandon(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res) {
  int * __restrict heap_ids = res.ids + q_idx * k;
  float * __restrict heap_dis = res.val + q_idx * k;

  f::heap_heapify<f::CMax<float, int>> (k, heap_dis, heap_ids);

  float bsfK = std::numeric_limits<float>::max();
  uint16_t * codes = mCodebook.data();
  for (int i = 0; i < mCodebook.rows(); i++) {
    float dist = 0;
    const float * luts = lut.data();

    const int ksub = lut.rows();
    int col;
    for (col=0; (col < mHighestSubs) && (dist < bsfK); col += 4) {
      float dism = 0;
      dism  = luts[*codes++]; luts += ksub;
      dism += luts[*codes++]; luts += ksub;
      dism += luts[*codes++]; luts += ksub;
      dism += luts[*codes++]; luts += ksub;
      dist += dism;
    }
    codes += (mHighestSubs-col);
    
    if (f::CMax<float, int>::cmp(heap_dis[0], dist)) {
      f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
      f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, i);
      bsfK = heap_dis[0];
    }

  }

  f::heap_reorder<f::CMax<float, int>> (k, heap_dis, heap_ids);
}

void VAQ::searchHeap(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res) {
  int * __restrict heap_ids = res.ids + q_idx * k;
  float * __restrict heap_dis = res.val + q_idx * k;

  f::heap_heapify<f::CMax<float, int>> (k, heap_dis, heap_ids);

  uint16_t * codes = mCodebook.data();
  for (int i = 0; i < mCodebook.rows(); i++) {
    float dist = 0;
    const float * luts = lut.data();

    const int ksub = lut.rows();
    for (int col=0; col < mHighestSubs; col += 4) {
      float dism = 0;
      dism  = luts[*codes++]; luts += ksub;
      dism += luts[*codes++]; luts += ksub;
      dism += luts[*codes++]; luts += ksub;
      dism += luts[*codes++]; luts += ksub;
      dist += dism;
    }
    
    if (f::CMax<float, int>::cmp(heap_dis[0], dist)) {
      f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
      f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, i);
    }

  }

  f::heap_reorder<f::CMax<float, int>> (k, heap_dis, heap_ids);
}

void VAQ::searchSort(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res) {
  std::vector<float> pairsDist(mCodebook.rows(), 0);
  
  auto codes = mCodebook.data();
  for (int i = 0; i < mCodebook.rows(); i++) {
    for (int col=0; col < mHighestSubs; col++) {
      auto luts = lut.data() + lut.rows() * col;
      pairsDist[i] += luts[codes[col]];
    }
    pairsDist[i] = std::sqrt(pairsDist[i]);

    codes += mCodebook.cols();
  }
  
  int res_offset = q_idx * k;
  KNNFromDists(pairsDist.data(), mCodebook.rows(), k, res, res_offset);
}

std::vector<IdxDistPairInt16> VAQ::searchFast(LUTType &lut, const int k) {
  constexpr int batchSize = 32;

  // create smallLUT from lut
  auto quantizeLUT = [this](const LUTType &oriLUT) -> SmallLUTType {
    return smallQuantize(
      (oriLUT.rowwise() - mOffsets).array().max(0).matrix(),
      mScale
    );
  };

  SmallLUTType smallLUT = quantizeLUT(lut);

  int batchNum = mCodebookCMajor.rows() / batchSize;
  if (mCodebookCMajor.rows() % batchSize > 0) {
    batchNum += 1;
  }

  std::vector<int16_t> pairsDist(mCodebookCMajor.rows(), 0);
  auto lutPtr = smallLUT.data();
  auto codebookPtr = mCodebookCMajor.data();
  for (int s = 0; s < mHighestSubs; s++) {
    auto distsItr = pairsDist.begin();
    uint8_t table[32] = {0};  // 256bit simd register
    std::copy(lutPtr, lutPtr + mCentroidsNum[s], std::begin(table));

    for (int b = 0; b < batchNum; b++) {
      uint8_t indices[batchSize];
      int16_t currdists[batchSize / 2];

      std::copy(codebookPtr, codebookPtr + batchSize, indices);

      __m256i vDists = ShuffleAVX2(*(__m256i *)&table, *(__m256i *)&indices);
      __m128i lowerDist = _mm256_extracti128_si256(vDists, 0);
      __m128i higherDist = _mm256_extracti128_si256(vDists, 1);
      
      // add the lower half first
      std::copy(distsItr, distsItr + (batchSize / 2), std::begin(currdists));
      __m256i tempDists = _mm256_cvtepu8_epi16(lowerDist);
      (* (__m256i *)(& currdists)) = _mm256_adds_epi16((* (__m256i *)(& currdists)), tempDists);
      std::copy(std::begin(currdists), std::end(currdists), distsItr);
      distsItr += (batchSize / 2);

      // add the higher half
      std::copy(distsItr, distsItr + (batchSize / 2), std::begin(currdists));
      tempDists = _mm256_cvtepu8_epi16(higherDist);
      (* (__m256i *)(& currdists)) = _mm256_adds_epi16((* (__m256i *)(& currdists)), tempDists);
      std::copy(std::begin(currdists), std::end(currdists), distsItr);
      distsItr += (batchSize / 2);

      codebookPtr += batchSize;
    }
    
    lutPtr += smallLUT.rows();
  }
  pairsDist.resize(mXTrainRows);

  return KNNFromDists(pairsDist.data(), pairsDist.size(), k);
}

// std::vector<IdxDistPairInt16> VAQ::searchFast(LUTType &lut, const int k) {
//   constexpr int batchSize = 16;

//   // create smallLUT from lut
//   auto quantizeLUT = [this](const LUTType &oriLUT) -> SmallLUTType {
//     return smallQuantize(
//       (oriLUT.rowwise() - mOffsets).array().max(0).matrix(),
//       mScale
//     );
//   };

//   SmallLUTType smallLUT = quantizeLUT(lut);

//   int batchNum = mCodebookCMajor.rows() / batchSize;
//   if (mCodebookCMajor.rows() % batchSize > 0) {
//     batchNum += 1;
//   }

//   std::vector<int16_t> pairsDist(mCodebookCMajor.rows(), 0);
//   auto lutPtr = smallLUT.data();
//   auto codebookPtr = mCodebookCMajor.data();
//   for (int s = 0; s < mHighestSubs; s++) {
//     auto distsItr = pairsDist.begin();
//     uint8_t table[16] = {0};  // 128bit simd register
//     std::copy(lutPtr, lutPtr + mCentroidsNum[s], std::begin(table));

//     for (int b = 0; b < batchNum; b++) {
//       uint8_t indices[batchSize];
//       int16_t currdists[batchSize];

//       std::copy(codebookPtr, codebookPtr + batchSize, indices);

//       __m128i vDists = _mm_shuffle_epi8(*(__m128i *)&table, *(__m128i *)&indices);
      
//       std::copy(distsItr, distsItr + batchSize, std::begin(currdists));
//       __m256i tempDists = _mm256_cvtepu8_epi16(vDists);
//       (* (__m256i *)(& currdists)) = _mm256_adds_epi16((* (__m256i *)(& currdists)), tempDists);
//       std::copy(std::begin(currdists), std::end(currdists), distsItr);
//       distsItr += batchSize;

//       codebookPtr += batchSize;
//     }
    
//     lutPtr += smallLUT.rows();
//   }
//   pairsDist.resize(mXTrainPCA.rows());


//   return KNNFromDists(pairsDist.data(), pairsDist.size(), k);
// }

void VAQ::searchFast2(LUTType &lut, const int k, int q_idx, f::float_maxheap_t &res) {
  constexpr int batchSize = 8; 
  int batchNum = mCodebookCMajor16.rows() / batchSize;
  if (mCodebookCMajor16.rows() % batchSize > 0) {
    batchNum += 1;
  }

  std::vector<float> pairsDist(mCodebookCMajor16.rows(), 0);

  auto lutPtr = lut.data();
  auto codebookPtr = mCodebookCMajor16.data();

  for (int s=0; s < mHighestSubs; s++) {
    auto distItr = pairsDist.begin();
    for (int b=0; b<batchNum; b++) {

      uint16_t indices[batchSize];
      std::copy(codebookPtr, codebookPtr + batchSize, std::begin(indices));
      __m256i vIndices = _mm256_cvtepu16_epi32((*(__m128i *) &indices));

      __m256 vResults = _mm256_i32gather_ps(lutPtr, vIndices, 4);
      
      float dists[batchSize];
      std::copy(distItr, distItr + batchSize, std::begin(dists));
      (* (__m256 *)(& dists)) = _mm256_add_ps(*(__m256 *)& dists, vResults);
      
      std::copy(std::begin(dists), std::end(dists), distItr);

      distItr += batchSize;
      codebookPtr += batchSize;
    }
    lutPtr += lut.rows();
  }
  
  int res_offset = q_idx * k;
  KNNFromDists(pairsDist.data(), pairsDist.size(), k, res, res_offset);
}

void VAQ::searchFast3(LUTType &lutGather, const int k, int q_idx, f::float_maxheap_t &res) {
  constexpr int gatherBatchSize = 8;
  constexpr int shuffleBatchSize = 32;

  int paddedRowSize;
  if (mStartShufIdx == mHighestSubs) {
    paddedRowSize = mCodebookCMajor16.rows();
  } else if (mStartShufIdx == 0) {
    paddedRowSize = mCodebookCMajor.rows();
  } else {
    paddedRowSize = mCodebookCMajor.rows();
  } 
  std::vector<float> pairsDist(paddedRowSize, 0);

  if (mStartShufIdx > 0) {
    // 8 batch size
    int gatherBatchNum = mCodebookCMajor16.rows() / gatherBatchSize;
    if (mCodebookCMajor16.rows() % gatherBatchSize > 0) {
      gatherBatchNum += 1;
    }
    auto lutPtr = lutGather.data();
    auto codebookPtr = mCodebookCMajor16.data();

    for (int s=0; s < mStartShufIdx; s++) {
      auto distItr = pairsDist.begin();
      for (int b=0; b<gatherBatchNum; b++) {

        uint16_t indices[gatherBatchSize];
        std::copy(codebookPtr, codebookPtr + gatherBatchSize, std::begin(indices));
        __m256i vIndices = _mm256_cvtepu16_epi32((*(__m128i *) &indices));

        __m256 vResults = _mm256_i32gather_ps(lutPtr, vIndices, 4);
        
        float dists[gatherBatchSize];
        std::copy(distItr, distItr + gatherBatchSize, std::begin(dists));
        (* (__m256 *)(& dists)) = _mm256_add_ps(*(__m256 *)& dists, vResults);
        
        std::copy(std::begin(dists), std::end(dists), distItr);

        distItr += gatherBatchSize;
        codebookPtr += gatherBatchSize;
      }
      lutPtr += lutGather.rows();
    }
  }

  if (mStartShufIdx < mHighestSubs) {
    // create smallLUT from portion of lut
    SmallLUTType lutShuffle = smallQuantize(
      (lutGather.block(0, mStartShufIdx, lutGather.rows(), lutGather.cols() - mStartShufIdx).array().max(0).matrix()),
      mScale
    );

    // 32 batch size
    int shuffleBatchNum = mCodebookCMajor.rows() / shuffleBatchSize;
    if (mCodebookCMajor.rows() % shuffleBatchSize > 0) {
      shuffleBatchNum += 1;
    }

    auto lutPtr = lutShuffle.data();
    auto codebookPtr = mCodebookCMajor.data();
    for (int s = mStartShufIdx; s < mHighestSubs; s++) {
      auto distsItr = pairsDist.begin();
      uint8_t table[32] = {0};  // 256bit simd register
      std::copy(lutPtr, lutPtr + mCentroidsNum[s], std::begin(table));

      __m256 vScale = _mm256_set1_ps(mScale(s-mStartShufIdx));
      __m256 vOffset = _mm256_set1_ps(mOffsets(s-mStartShufIdx));

      for (int b = 0; b < shuffleBatchNum; b++) {
        uint8_t indices[shuffleBatchSize];
        float currdists[shuffleBatchSize / 4];

        std::copy(codebookPtr, codebookPtr + shuffleBatchSize, indices);

        __m256i vDists = ShuffleAVX2(*(__m256i *)&table, *(__m256i *)&indices);
        __m128i lowerDist = _mm256_extracti128_si256(vDists, 0);
        __m128i higherDist = _mm256_extracti128_si256(vDists, 1);

        // 1
        __m256 halfDistF = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lowerDist));
        __m256 distsInvQuantized = fma(halfDistF, vScale, vOffset);
        std::copy(distsItr, distsItr + (shuffleBatchSize / 4), std::begin(currdists));
        (* (__m256 *)(& currdists)) = _mm256_add_ps((* (__m256 *)(& currdists)), distsInvQuantized);
        std::copy(std::begin(currdists), std::end(currdists), distsItr);
        distsItr += (shuffleBatchSize / 4);
        
        // 2
        halfDistF = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(lowerDist, 8)));
        distsInvQuantized = fma(halfDistF, vScale, vOffset);
        std::copy(distsItr, distsItr + (shuffleBatchSize / 4), std::begin(currdists));
        (* (__m256 *)(& currdists)) = _mm256_add_ps((* (__m256 *)(& currdists)), distsInvQuantized);
        std::copy(std::begin(currdists), std::end(currdists), distsItr);
        distsItr += (shuffleBatchSize / 4);
        
        // 3
        halfDistF = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(higherDist));
        distsInvQuantized = fma(halfDistF, vScale, vOffset);
        std::copy(distsItr, distsItr + (shuffleBatchSize / 4), std::begin(currdists));
        (* (__m256 *)(& currdists)) = _mm256_add_ps((* (__m256 *)(& currdists)), distsInvQuantized);
        std::copy(std::begin(currdists), std::end(currdists), distsItr);
        distsItr += (shuffleBatchSize / 4);
        
        // 4
        halfDistF = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(higherDist, 8)));
        distsInvQuantized = fma(halfDistF, vScale, vOffset);
        std::copy(distsItr, distsItr + (shuffleBatchSize / 4), std::begin(currdists));
        (* (__m256 *)(& currdists)) = _mm256_add_ps((* (__m256 *)(& currdists)), distsInvQuantized);
        std::copy(std::begin(currdists), std::end(currdists), distsItr);
        distsItr += (shuffleBatchSize / 4);

        codebookPtr += shuffleBatchSize;
      }
      
      lutPtr += lutShuffle.rows();
    }
  }

  int res_offset = q_idx * k;
  KNNFromDists(pairsDist.data(), pairsDist.size(), k, res, res_offset);
}
