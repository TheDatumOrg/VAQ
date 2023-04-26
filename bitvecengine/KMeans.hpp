#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <vector>
#include <limits>
#include <iostream>
#include <cmath>
#include <omp.h>

#include <Eigen/Core>

#include "utils/Types.hpp"
#include "utils/DistanceFunctions.hpp"
#include "utils/Random.hpp"

class KMeans {
private:
  int k, max_iter;
  Matrixf init;
  bool isSampling;
public:
  enum Seed {
    RANDOM, // random rows for initial centroids
    CUSTOM,
    KMeansPP
  };
  
  Matrixf last_centroids;

  /**
   * @brief Construct a new KMeans object
   * 
   * @param n_cluster K
   * @param seed seed type
   * @param init must be supplied if seed == CUSTOM, contains k index of instance in dataset 
   */
  KMeans(int n_cluster, int max_iter=25, bool isSampling=true, Seed seed = Seed::RANDOM) :
    k(n_cluster), max_iter(max_iter), isSampling(isSampling), seed(seed) {
      srand(time(NULL));
    }

  std::vector<int> fit(Matrixf X) {
    std::cout << "Clustering using KMeans (K = " << this->k << ")" << std::endl;
    const int nrows = X.size();
    const int dimen = X[0].size();
    
    Matrixf means(this->k);

    // initialize means
    if (seed == Seed::RANDOM) {
      for (int i=0; i<this->k; i++) {
        means[i] = X[rand() % nrows];
      }
    } else if (seed == Seed::CUSTOM) {
      for (int i=0; i<this->k; i++) {
        means[i] = init[i];
      }
    }

    bool centroids_changed = true;
    std::vector<int> belongs_to(nrows);
    int iter_count = 0;
    while (centroids_changed && iter_count < this->max_iter) {     
      Matrixf new_centroids(this->k, std::vector<float>(dimen, 0));
      std::vector<int> count_centroids(this->k, 0);
      // assign cluster & new centroids
      for (int i=0; i<nrows; i++) {
        float dist_min = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j=0; j<this->k; j++) {
          float dist = euclideanDist(X[i], means[j]);
          if (dist < dist_min) {
            dist_min = dist;
            min_idx = j;
          }
        }
        belongs_to[i] = min_idx;
        for (int j=0; j<dimen; j++) {
          new_centroids[min_idx][j] += X[i][j];
        }
        count_centroids[min_idx]++;
      }

      centroids_changed = false;
      for (int i=0; i<this->k; i++) {
        for (int j=0; j<dimen; j++) {
          new_centroids[i][j] /= count_centroids[i];
        }
        if (!(new_centroids[i] == means[i])) {
          centroids_changed = true;
          means[i] = new_centroids[i];
        }
      }
      iter_count++;
    }
    if (iter_count >= this->max_iter) {
      std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
    } else {
      std::cout << "iter_count: " << iter_count << std::endl;
    }

    for (int i=0; i<this->k; i++) {
      this->last_centroids.push_back(means[i]);
    }

    return belongs_to;
  }

  std::vector<int> fitParallel(Matrixf X, int thread=20) {
    std::cout << "Clustering using KMeans (K = " << this->k << ")" << std::endl;
    const int nrows = X.size();
    const int dimen = X[0].size();
    
    Matrixf means(this->k);

    // initialize means
    if (seed == Seed::RANDOM) {
      for (int i=0; i<this->k; i++) {
        means[i] = X[rand() % nrows];
      }
    } else if (seed == Seed::CUSTOM) {
      for (int i=0; i<this->k; i++) {
        means[i] = init[i];
      }
    }

    bool centroids_changed = true;
    std::vector<int> belongs_to(nrows);
    int iter_count = 0;
    while (centroids_changed && iter_count < this->max_iter) {     
      Matrixf new_centroids(this->k, std::vector<float>(dimen, 0));
      std::vector<int> count_centroids(this->k, 0);
      // assign cluster & new centroids
      #pragma omp parallel num_threads(thread)
      {
        Matrixf new_centroids_private(this->k, std::vector<float>(dimen, 0));
        std::vector<int> count_centroids_private(this->k, 0);

        #pragma omp for nowait schedule(static)
        for (int i=0; i<nrows; i++) {
          float dist_min = std::numeric_limits<float>::max();
          int min_idx = -1;
          for (int j=0; j<this->k; j++) {
            float dist = euclideanDist(X[i], means[j]);
            if (dist < dist_min) {
              dist_min = dist;
              min_idx = j;
            }
          }
          belongs_to[i] = min_idx;
          for (int j=0; j<dimen; j++) {
            new_centroids_private[min_idx][j] += X[i][j];
          }
          count_centroids_private[min_idx]++;
        }

        #pragma omp for schedule(static) ordered
        for (int i=0; i<nrows; i++) {
          #pragma omp ordered
          {
            for (int j=0; j<dimen; j++) {
              new_centroids[belongs_to[i]][j] += new_centroids_private[belongs_to[i]][j];
            }
            count_centroids[belongs_to[i]]++;
          }
        }
      }

      centroids_changed = false;
      for (int i=0; i<this->k; i++) {
        for (int j=0; j<dimen; j++) {
          new_centroids[i][j] /= count_centroids[i];
        }
        if (!(new_centroids[i] == means[i])) {
          centroids_changed = true;
          means[i] = new_centroids[i];
        }
      }
      iter_count++;
    }
    if (iter_count >= this->max_iter) {
      std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
    } else {
      std::cout << "iter_count: " << iter_count << std::endl;
    }

    for (int i=0; i<this->k; i++) {
      this->last_centroids.push_back(means[i]);
    }

    return belongs_to;
  }

  std::vector<int> fastFit(Matrixf X, int batch_size=100, float tol=1e-4) {
    std::cout << "Clustering using MiniBatchKMeans (K = " << this->k << ")" << std::endl;
    const int nrows = X.size();
    const int dimen = X[0].size();
    
    Matrixf means(this->k);

    // initialize means
    if (seed == Seed::RANDOM) {
      for (int i=0; i<this->k; i++) {
        means[i] = X[rand() % nrows];
      }
    } else if (seed == Seed::CUSTOM) {
      for (int i=0; i<this->k; i++) {
        means[i] = init[i];
      }
    }

    std::vector<int> cluster_counts(batch_size, 0);
    float total_error = 1;
    int iter;
    for (iter=0; iter<this->max_iter && total_error > tol; iter++) {
      // select batch
      Matrixf batch(batch_size);
      std::vector<int> clusters(batch_size);
      for (int i=0; i<batch_size; i++) {
        batch[i] = X[rand() % nrows];
        
        float min_dist = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int m_iter=0; m_iter<this->k; m_iter++) {
          float dist = euclideanDist(batch[i], means[m_iter]);
          if (dist < min_dist) {
            min_dist = dist;
            min_idx = m_iter;
          }
        }
        clusters[i] = min_idx;
      }
      
      // save means before update
      Matrixf centroids(this->k);
      for (int i=0; i<this->k; i++) {
        centroids[i] = means[i];
      }

      // update means
      float eta;
      int idx;
      for (int b_iter=0; b_iter<batch_size; b_iter++) {
        idx = clusters[b_iter];
        cluster_counts[idx]++;
        eta = 1.0f / cluster_counts[idx];
        
        for (int d_iter=0; d_iter < dimen; d_iter++) {
          means[idx][d_iter] = (1.0f - eta) * means[idx][d_iter] + eta * batch[b_iter][d_iter];
        }
      }

      // calculate error
      total_error = 0;
      for (int i=0; i<this->k; i++) {
        // total_error += euclideanDist(centroids[i], means[i]);
        total_error += manhattanDist(centroids[i], means[i]);
      }
    }
    if (iter >= this->max_iter) {
      std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
    } else {
      std::cout << "iter_count: " << iter << std::endl;
    }
    std::cout << "last error: " << total_error << std::endl; 
    
    std::vector<int> belongs_to(nrows);
    for (int i=0; i<nrows; i++) {
      float dist_min = std::numeric_limits<float>::max();
      int min_idx = -1;
      for (int j=0; j<this->k; j++) {
        float dist = euclideanDist(X[i], means[j]);
        if (dist < dist_min) {
          dist_min = dist;
          min_idx = j;
        }
      }
      belongs_to[i] = min_idx;
    }

    for (int i=0; i<this->k; i++) {
      this->last_centroids.push_back(means[i]);
    }

    return belongs_to;
  }

  static CentroidsMatType staticFit(const Eigen::MatrixXf &X, int k, int max_iter, int maxcentroids, bool verbose=false, Seed seedType = Seed::RANDOM) {
    if (verbose) {
      std::cout << "Clustering using KMeans (K = " << k << ")" << std::endl;
    }
    const int nrows = X.rows();
    const int dimen = X.cols();
    
    maxcentroids = k;
    CentroidsMatType means(maxcentroids, dimen);

    // initialize means
    if (seedType == Seed::RANDOM) {
      for (int i=0; i<k; i++) {
        means.row(i) = X.row(rand() % nrows);
      }
    } else if (seedType == Seed::KMeansPP) {
      // select the first centroid randomly
      int centroidIdx = 0;
      means.row(centroidIdx) = X.row(rand() % nrows);
      Eigen::VectorXf dist(nrows);
      for (centroidIdx = centroidIdx + 1; centroidIdx < k ; centroidIdx++) {
        dist.setZero();
        for (int i=0; i<nrows; i++) {
          float minDist = std::numeric_limits<float>::max();
          
          const int currentCentroidSize = centroidIdx;
          for (int j=0; j<currentCentroidSize; j++) {
            float tempDist = (X.row(i) - means.row(j)).squaredNorm();
            if (tempDist < minDist) {
              minDist = tempDist;
            }
          }

          dist(i) = minDist;
        }

        Eigen::VectorXf::Index nextCentroidIdx;
        dist.maxCoeff(&nextCentroidIdx);
        means.row(centroidIdx) = X.row(nextCentroidIdx);
      }
    }

    auto euclideanDistance = [&means, &X, dimen](int xrow, int meanrow) -> float{
      float dist = (X.row(xrow) - means.row(meanrow)).squaredNorm();
      return sqrt(dist);
    };

    bool centroids_changed = true;
    std::vector<int> belongs_to(nrows);
    int iter_count = 0;
    while (centroids_changed && iter_count < max_iter) {     
      CentroidsMatType new_centroids(maxcentroids, dimen);
      new_centroids.setZero();
      std::vector<int> count_centroids(k, 0);
      // assign cluster & new centroids
      for (int i=0; i<nrows; i++) {
        float dist_min = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j=0; j<k; j++) {
          float dist = euclideanDistance(i, j);
          if (dist < dist_min) {
            dist_min = dist;
            min_idx = j;
          }
        }
        belongs_to[i] = min_idx;
        new_centroids.row(min_idx) += X.row(i);
        count_centroids[min_idx]++;
      }

      centroids_changed = false;
      for (int i=0; i<k; i++) {
        new_centroids.row(i) /= count_centroids[i];
        if (!(new_centroids.row(i) == means.row(i))) {
          centroids_changed = true;
          means.row(i) = new_centroids.row(i);
        }
      }
      iter_count++;
    }
    if (verbose) {
      if (iter_count >= max_iter) {
        std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
      } else {
        std::cout << "iter_count: " << iter_count << std::endl;
      }
    }

    return means;
  }
  
  static CentroidsMatType staticFitSampling(const Eigen::MatrixXf &XTrain, int k, int max_iter, int maxcentroids, bool verbose=false, Seed seedType = Seed::RANDOM) {
    const int MaxPointsPerCentroids = 256;
    // const int MaxPointsPerCentroids = 64;

    Eigen::MatrixXf X;
    if (XTrain.rows() > MaxPointsPerCentroids * k) {
      // sample the dataset
      X.resize(MaxPointsPerCentroids * k, XTrain.cols());
      std::vector<int> perm(XTrain.rows());
      randomPermutation(perm);
      for (int i=0; i<X.rows(); i++) {
        X.row(i) = XTrain.row(perm[i]);
      }
    } else {
      X = XTrain;
    }

    if (verbose) {
      std::cout << "Clustering using KMeans with sampling (K = " << k << ")" << std::endl;
      std::cout << "Sample size = " << X.rows() << std::endl;
    }
    const int nrows = X.rows();
    const int dimen = X.cols();
    
    maxcentroids = k;
    CentroidsMatType means(maxcentroids, dimen);

    // initialize means
    if (seedType == Seed::RANDOM) {
      for (int i=0; i<k; i++) {
        means.row(i) = X.row(rand() % nrows);
      }
    } else if (seedType == Seed::KMeansPP) {
      // select the first centroid randomly
      int centroidIdx = 0;
      means.row(centroidIdx) = X.row(rand() % nrows);
      Eigen::VectorXf dist(nrows);
      for (centroidIdx = centroidIdx + 1; centroidIdx < k ; centroidIdx++) {
        dist.setZero();
        for (int i=0; i<nrows; i++) {
          float minDist = std::numeric_limits<float>::max();
          
          const int currentCentroidSize = centroidIdx;
          for (int j=0; j<currentCentroidSize; j++) {
            float tempDist = (X.row(i) - means.row(j)).squaredNorm();
            if (tempDist < minDist) {
              minDist = tempDist;
            }
          }

          dist(i) = minDist;
        }

        Eigen::VectorXf::Index nextCentroidIdx;
        dist.maxCoeff(&nextCentroidIdx);
        means.row(centroidIdx) = X.row(nextCentroidIdx);
      }
    }

    auto euclideanDistance = [&means, &X, dimen](int xrow, int meanrow) -> float{
      float dist = (X.row(xrow) - means.row(meanrow)).squaredNorm();
      return sqrt(dist);
    };

    bool centroids_changed = true;
    std::vector<int> belongs_to(nrows);
    int iter_count = 0;
    while (centroids_changed && iter_count < max_iter) {     
      CentroidsMatType new_centroids(maxcentroids, dimen);
      new_centroids.setZero();
      std::vector<int> count_centroids(k, 0);
      // assign cluster & new centroids
      for (int i=0; i<nrows; i++) {
        float dist_min = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j=0; j<k; j++) {
          float dist = euclideanDistance(i, j);
          if (dist < dist_min) {
            dist_min = dist;
            min_idx = j;
          }
        }
        belongs_to[i] = min_idx;
        new_centroids.row(min_idx) += X.row(i);
        count_centroids[min_idx]++;
      }

      centroids_changed = false;
      for (int i=0; i<k; i++) {
        new_centroids.row(i) /= count_centroids[i];
        if (!(new_centroids.row(i) == means.row(i))) {
          centroids_changed = true;
          means.row(i) = new_centroids.row(i);
        }
      }
      iter_count++;
    }
    if (verbose) {
      if (iter_count >= max_iter) {
        std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
      } else {
        std::cout << "iter_count: " << iter_count << std::endl;
      }
    }

    return means;
  }

  static CentroidsMatType staticFitSampling(const RowMatrixXf &XTrain, int k, int max_iter, bool verbose=false, Seed seedType = Seed::RANDOM) {
    // const int MaxPointsPerCentroids = 256;
    const int sampleSize = std::max(k * 256, 256*256);
    // const int MaxPointsPerCentroids = 64;

    RowMatrixXf X;
    if (XTrain.rows() > sampleSize) {
      // sample the dataset
      X.resize(sampleSize, XTrain.cols());
      std::vector<int> perm(XTrain.rows());
      randomPermutation(perm);
      for (int i=0; i<X.rows(); i++) {
        X.row(i) = XTrain.row(perm[i]);
      }
    } else {
      X = XTrain;
    }

    if (verbose) {
      std::cout << "Clustering using KMeans with sampling (K = " << k << ")" << std::endl;
      std::cout << "Sample size = " << X.rows() << std::endl;
    }
    const int nrows = X.rows();
    const int dimen = X.cols();
    
    CentroidsMatType means(k, dimen);

    // initialize means
    if (seedType == Seed::RANDOM) {
      std::vector<int> perm(nrows);
      randomPermutation(perm);
      for (int i=0; i<k; i++) {
        means.row(i) = X.row(perm[i]);
      }
    } else if (seedType == Seed::KMeansPP) {
      // select the first centroid randomly
      int centroidIdx = 0;
      means.row(centroidIdx) = X.row(rand() % nrows);
      Eigen::VectorXf dist(nrows);
      for (centroidIdx = centroidIdx + 1; centroidIdx < k ; centroidIdx++) {
        dist.setZero();
        for (int i=0; i<nrows; i++) {
          float minDist = std::numeric_limits<float>::max();
          
          const int currentCentroidSize = centroidIdx;
          for (int j=0; j<currentCentroidSize; j++) {
            float tempDist = (X.row(i) - means.row(j)).squaredNorm();
            if (tempDist < minDist) {
              minDist = tempDist;
            }
          }

          dist(i) = minDist;
        }

        Eigen::VectorXf::Index nextCentroidIdx;
        dist.maxCoeff(&nextCentroidIdx);
        means.row(centroidIdx) = X.row(nextCentroidIdx);
      }
    }

    auto euclideanDistance = [&means, &X, dimen](int xrow, int meanrow) -> float{
      float dist = (X.row(xrow) - means.row(meanrow)).squaredNorm();
      return sqrt(dist);
    };
    constexpr int ompThread = 2;

    bool centroids_changed = true;
    std::vector<int> belongs_to(nrows);
    int iter_count = 0;
    while (centroids_changed && iter_count < max_iter) {     
      CentroidsMatType new_centroids(k, dimen);
      new_centroids.setZero();
      std::vector<int> count_centroids(k, 0);
      // assign cluster & new centroids
      #pragma omp parallel num_threads(ompThread)
      {
        CentroidsMatType privatenew_centroids(k, dimen);
        privatenew_centroids.setZero();
        std::vector<int> privatecount_centroids(k, 0);

        #pragma omp for nowait schedule(static)
        for (int i=0; i<nrows; i++) {
          float dist_min = std::numeric_limits<float>::max();
          int min_idx = -1;
          for (int j=0; j<k; j++) {
            float dist = euclideanDistance(i, j);
            if (dist < dist_min) {
              dist_min = dist;
              min_idx = j;
            }
          }
          belongs_to[i] = min_idx;
          privatenew_centroids.row(min_idx) += X.row(i);
          privatecount_centroids[min_idx]++;
        }

        const int num_threads = omp_get_num_threads();
        #pragma omp for schedule(static) ordered
        for (int i=0; i<num_threads; i++) {
          #pragma omp ordered
          {
            for (int j=0; j<k; j++) {
              new_centroids.row(j) += privatenew_centroids.row(j);
              count_centroids[j] += privatecount_centroids[j];
            }
          }
        }
      }

      centroids_changed = false;
      for (int i=0; i<k; i++) {
        new_centroids.row(i) /= count_centroids[i];
        if (!(new_centroids.row(i) == means.row(i))) {
          centroids_changed = true;
          means.row(i) = new_centroids.row(i);
        }
      }
      iter_count++;
    }
    if (verbose) {
      if (iter_count >= max_iter) {
        std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
      } else {
        std::cout << "iter_count: " << iter_count << std::endl;
      }
    }

    return means;
  }

  static CentroidsMatType staticFitCodebook(const CodebookType& codebookTrain, const CentroidsPerSubsType &centroidsPerDim, const int firstSubsNum, const int k, const int max_iter=100, bool verbose=false) {
    const int MaxVectorPerCentroids = 256;

    RowMatrixXf X;
    int totalOriginalDim = 0;
    for (int i=0; i<firstSubsNum; i++) {
      totalOriginalDim += centroidsPerDim[i].cols();
    }
    if (codebookTrain.rows() > MaxVectorPerCentroids * k) {
      // sample the training set
      X.resize(MaxVectorPerCentroids * k, totalOriginalDim);
      std::vector<int> perm(codebookTrain.rows());
      randomPermutation(perm);
      for (int i=0; i<X.rows(); i++) {
        int jDim = 0;
        for (int j=0; j<firstSubsNum; j++) {
          X.block(i, jDim, 1, centroidsPerDim[j].cols()) = centroidsPerDim[j].row(codebookTrain(perm[i], j));
          jDim += centroidsPerDim[j].cols();
        }
      }
    } else {
      X.resize(codebookTrain.rows(), totalOriginalDim);
      for (int i=0; i<X.rows(); i++) {
        int jDim = 0;
        for (int j=0; j<firstSubsNum; j++) {
          X.block(i, jDim, 1, centroidsPerDim[j].cols()) = centroidsPerDim[j].row(codebookTrain(i, j));
          jDim += centroidsPerDim[j].cols();
        }
      }
    }

    CentroidsMatType means = KMeans::staticFitSampling(X, k, max_iter, verbose);

    return means;
  }

  static CentroidsMatType staticFastFit(const Eigen::MatrixXf &X, int k, int max_iter=100, int batch_size=100, float tol=0.0, int maxcentroids=256, bool verbose=false) {
    if (verbose) {
      std::cout << "Clustering using MiniBatchKMeans (K = " << k << ")" << std::endl;
    }
    const int nrows = X.rows();
    const int dimen = X.cols();
    
    CentroidsMatType means(maxcentroids, dimen);

    // initialize means
    for (int i=0; i<k; i++) {
      means.row(i) = X.row(rand() % nrows);
    }

    std::vector<int> cluster_counts(batch_size, 0);
    float total_error = 1;
    int iter;
    for (iter=0; iter<max_iter && total_error > tol; iter++) {
      // select batch
      Eigen::MatrixXf batch(batch_size, dimen);
      std::vector<int> clusters(batch_size);
      for (int i=0; i<batch_size; i++) {
        batch.row(i) = X.row(rand() % nrows);

        auto euclideanDistance = [&means, &batch, dimen](int batchrow, int meanrow) -> float{
          float dist = 0;
          for (int i=0; i<dimen; i++) {
            float diff = batch(batchrow, i)-means(meanrow, i);
            dist += diff * diff;
          }

          return sqrt(dist);
        };
        
        float min_dist = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int m_iter=0; m_iter<k; m_iter++) {
          float dist = euclideanDistance(i, m_iter);
          if (dist < min_dist) {
            min_dist = dist;
            min_idx = m_iter;
          }
        }
        clusters[i] = min_idx;
      }
      
      // save means before update
      Eigen::MatrixXf centroids(maxcentroids, dimen);
      for (int i=0; i<k; i++) {
        centroids.row(i) = means.row(i);
      }

      // update means
      float eta;
      int idx;
      for (int b_iter=0; b_iter<batch_size; b_iter++) {
        idx = clusters[b_iter];
        cluster_counts[idx]++;
        eta = 1.0f / cluster_counts[idx];

        means.row(idx) = (1.0f - eta) * means.row(idx) + (eta * batch.row(b_iter));
      }

      auto euclideanDistanceErr = [&means, &centroids, dimen](int idx) -> float{
        float dist = 0;
        for (int i=0; i<dimen; i++) {
          float diff = centroids(idx, i)-means(idx, i);
          dist += diff * diff;
        }

        return sqrt(dist);
      };

      // calculate error
      total_error = 0;
      for (int i=0; i<k; i++) {
        total_error += euclideanDistanceErr(i);
      }
    }
    if (verbose) {
      if (iter >= max_iter) {
        std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
      } else {
        std::cout << "iter_count: " << iter << std::endl;
      }
      std::cout << "last error: " << total_error << std::endl; 
    }
    
    return means;
  }

  static Eigen::VectorXi staticFitIndex(const std::vector<float>& X, const int k, const int max_iter, bool verbose=false) {
    const int nrows = X.size();
    
    std::vector<float> means(k);

    // initialize means
    std::vector<int> perm(X.size());
    randomPermutation(perm);
    for (int i=0; i<k; i++) {
      means[i] = X[perm[i]];
      std::cout << perm[i] << ", ";
    }
    std::cout << std::endl;

    bool centroids_changed = true;
    Eigen::VectorXi belongs_to(nrows);
    std::vector<int> count_centroids(k, 0);
    int iter_count = 0;
    while (centroids_changed && iter_count <= max_iter) {     
      std::vector<float> new_centroids(k);
      std::fill(count_centroids.begin(), count_centroids.end(), 0);
      // assign cluster & new centroids
      for (int i=0; i<nrows; i++) {
        float dist_min = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j=0; j<k; j++) {
          float dist = std::abs(X[i] - means[j]);
          if (dist < dist_min) {
            dist_min = dist;
            min_idx = j;
          }
        }
        belongs_to(i) = min_idx;
        new_centroids[min_idx] += X[i];
        count_centroids[min_idx]++;
      }

      centroids_changed = false;
      for (int i=0; i<k; i++) {
        new_centroids[i] /= count_centroids[i];
        if (!(new_centroids[i] == means[i])) {
          centroids_changed = true;
          means[i] = new_centroids[i];
        }
      }
      iter_count++;
    }
    if (verbose) {
      if (iter_count >= max_iter) {
        std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
      } else {
        std::cout << "iter_count: " << iter_count << std::endl;
      }
    }

    return belongs_to;
  }

  static Eigen::VectorXi staticFitIndex(const Eigen::VectorXf& X, const int k, const int max_iter, bool verbose=false) {
    const int nrows = X.size();
    
    Eigen::VectorXf means(k);

    // initialize means
    std::vector<int> perm(X.size());
    randomPermutation(perm);
    for (int i=0; i<k; i++) {
      means(i) = X(perm[i]);
    }

    bool centroids_changed = true;
    Eigen::VectorXi belongs_to(nrows);
    std::vector<int> count_centroids(k, 0);
    int iter_count = 0;
    while (centroids_changed && iter_count <= max_iter) {     
      std::vector<float> new_centroids(k);
      std::fill(count_centroids.begin(), count_centroids.end(), 0);
      // assign cluster & new centroids
      for (int i=0; i<nrows; i++) {
        float dist_min = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j=0; j<k; j++) {
          float dist = std::abs(X(i) - means(j));
          if (dist < dist_min) {
            dist_min = dist;
            min_idx = j;
          }
        }
        belongs_to(i) = min_idx;
        new_centroids.at(min_idx) += X(i);
        count_centroids.at(min_idx) += 1;
      }

      centroids_changed = false;
      for (int i=0; i<k; i++) {
        new_centroids.at(i) /= count_centroids.at(i);
        if (!(new_centroids.at(i) == means(i))) {
          centroids_changed = true;
          means(i) = new_centroids.at(i);
        }
      }
      iter_count++;
    }
    if (verbose) {
      if (iter_count >= max_iter) {
        std::cout << "algorithm stops before fully converging (max_iter: " << max_iter << ")" << std::endl;
      } else {
        std::cout << "iter_count: " << iter_count << std::endl;
      }
    }

    return belongs_to;
  }

private:
  KMeans::Seed seed;

};

#endif
