#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <getopt.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "KMeans.hpp"
#include "utils/Types.hpp"
#include "utils/Experiment.hpp"
#include "utils/IO.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/DistanceFunctions.hpp"

template<class T>
double compute_sillhoute(const T &bv, const std::vector<int> &labels, int k) {
  const int nrows = bv.size();
  double sillhoute = 0;

  // group clusters
  std::vector<std::vector<int>> clusters(k, std::vector<int>());
  for (int i=0; i<nrows; i++) {
    clusters[labels[i]].push_back(i);
  }

  int c_idx = 0;
  for (const std::vector<int> &clus: clusters) {
    const int current_clus_size = clus.size();
    double a = 0, b_temp = 0, b = std::numeric_limits<double>::max();
    for (int i=0; i<current_clus_size-1; i++) {
      for (int j=i+1; j<current_clus_size; j++) {
        a += euclideanDist(bv[clus[i]], bv[clus[j]]);
      }
    }
    a /= (current_clus_size / 2.0) * (1 + current_clus_size);

    for (int i=0; i<k; i++) {
      if (c_idx == i) {
        continue;
      }
      for (int j=0; j<current_clus_size; j++) {
        for (int m=0; m<(int)clusters[i].size(); m++) {
          b_temp += euclideanDist(bv[clus[j]], bv[clusters[i][m]]);
        }
      }
      b_temp /= current_clus_size * clusters[i].size();
      if (b_temp < b) {
        b = b_temp;
      }
    }

    sillhoute += (b - a) / ((a > b) ? a : b);
  }

  sillhoute /= k;
  return sillhoute;
}

int main(int argc, char **argv) {
  int K = 100;
  int datasetSize = -1;
  int N = 0;
  int M = 32; // subvector length
  std::string dataset = "";
  std::string fileFormatOri = "fvecs";
  std::string centroidsFilepath = "";
  std::string mode = "simple";
  std::string seedType = "random";

  while (true) {
    static struct option long_options[] = {
      {"k", required_argument, 0, 'k'},
      {"timeseries-size", required_argument, 0, 't'},
      {"dataset", required_argument, 0, 'a'},
      {"dataset-size", required_argument, 0, 'd'},
      {"mode", required_argument, 0, 'm'},
      {"seed", required_argument, 0, 's'},
      {"m", required_argument, 0, 'o'},
      {"file-format-ori", required_argument, 0, 'z'},
      {"centroids-output", required_argument, 0, 'c'},
      {"help", no_argument, 0, '?'}
    };

    int option_index = 0;
    int c = getopt_long (argc, argv, "", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
      case 'k': K = std::atoi(optarg); break;
      case 't': N = std::atoi(optarg); break;
      case 'a': dataset = optarg; break;
      case 'd': datasetSize = std::atoi(optarg); break;
      case 'm': mode = optarg; break;
      case 's': seedType = optarg; break;
      case 'z': fileFormatOri = optarg; break;
      case 'c': centroidsFilepath = optarg; break;
      case 'o': M = std::atoi(optarg); break;
      case '?':
        std::cout << 
        "Usage:\n\
        \t--k XX \t\t\tK in K-nearest-neighbor\n\
        \t--timeseries-size XX \t\t\tThe number of dimension\n\
        \t--dataset XX \t\t\tThe path to the binary dataset file\n\
        \t--dataset-size XX \t\tThe number of time series to load\n\
        \t--mode: \n\
        \t--\tsimple\n\
        \t--\tfast\n\
        \t--\tparallel\n\
        \t--\tbitvec\n\
        \t--\tsimple_pca\n\
        \t--seed: \n\
        \t--\trandom\n\
        \t--\tkmeans++\n\
        \t--m\t\tNumber of SubVector\n\
        \t--help\n\n\
        \t--**********************EXAMPLES**********************\n\n\
        \t--./mainlearn --dataset XX --dataset-size XX --k 100  \n\n\
        \t--****************************************************\n" << std::endl;
        exit(0);
        break;
    }
  }

  KMeans km(K);
  Matrixf vms;
  int nAppendColumn = 0;
  if (mode == "subvec") {
    if ((float)N / M != N / M) {
      nAppendColumn = (std::round((float)N / M) * M) - N;
    }
  }
  Eigen::MatrixXf oriDataset(datasetSize, N + nAppendColumn);
  
  std::cout << "Read dataset" << std::endl;
  if (mode == "subvec" || mode == "simple_pca") {
    if (fileFormatOri == "ascii") {
      readOriginalFromExternal<true>(dataset, oriDataset, N, ',');
    } else if (fileFormatOri == "fvecs") {
      readFVecsFromExternal(dataset, oriDataset, N, datasetSize);
    } else if (fileFormatOri == "bin") {
      readFromExternalBin(dataset, oriDataset, N, datasetSize);
    }
  } else {
    if (fileFormatOri == "ascii") {
      readOriginalFromExternal<true>(dataset, vms, N);
    } else if (fileFormatOri == "fvecs") {
      readFVecsFromExternal(dataset, vms, N, datasetSize);
    } else if (fileFormatOri == "bin") {
      readFromExternalBin(dataset, vms, N, datasetSize);
    }
  }

  CentroidsPerSubsType centroidsPerDim(M);
  Eigen::MatrixXf centroidsEigen;
  
  std::cout << "fit KMeans" << std::endl;
  cputime_t start = timeNow(), end;
  std::vector<int> ret;
  if (mode == "simple") {
    ret = km.fit(vms);
  } else if (mode == "fast") {
    ret = km.fastFit(vms, 100000, 1e-5);
  } else if (mode == "parallel") {
    ret = km.fitParallel(vms);
  } else if (mode == "subvec") {
    // K = number of centroids per segment
    int subvectorLen = oriDataset.cols() / M;
    for (int i=0; i<M; i++) {
      centroidsPerDim[i] = KMeans::staticFit(
        oriDataset.block(0, i*subvectorLen, oriDataset.rows(), subvectorLen),
        K, 
        16, // MaxIter
        K, // max centroids per segment
        false
      );
    }
  } else if (mode == "simple_pca") {
    // Principal Component Analysis
    Eigen::EigenSolver<Eigen::MatrixXf> es(oriDataset.transpose() * oriDataset);
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
    Eigen::MatrixXcf eigenVectors(es.eigenvectors().rows(), es.eigenvectors().cols());
    for (int i = 0; i < eigenVectors.cols(); i++) {
      eigenVectors.col(i) = es.eigenvectors().col(eigvalueidx[i].idx);
    }

    // Project on eigenvectors
    Eigen::MatrixXf datasetPCA(oriDataset.rows(), oriDataset.cols());
    Eigen::MatrixXcf ZxV2 = oriDataset * eigenVectors;
    for (int i=0; i<ZxV2.rows(); i++) {
      for (int j=0; j<ZxV2.cols(); j++) {
        if (ZxV2(i, j).imag() != 0 || std::isnan(ZxV2(i, j).real()) || std::isinf(ZxV2(i, j).real())) {
          datasetPCA(i, j) = 0;
        } else {
          datasetPCA(i, j) = ZxV2(i, j).real();
        }
      }
    }

    KMeans::Seed seed = KMeans::Seed::RANDOM;
    if (seedType == "random") {
      seed = KMeans::Seed::RANDOM;
    } else if (seedType == "kmeans++") {
      seed = KMeans::Seed::KMeansPP;
    }

    centroidsEigen = KMeans::staticFit(
      datasetPCA, 
      K,
      100000,
      K,
      false,
      seed
    );
  }
  end = timeNow();
  double elapsed = durationS(start, end);
  std::cout << elapsed << " seconds" << '\n';

  std::cout << "writing out centroids" << std::endl;
  if (mode == "subvec") {
    writeCentroidsPerDimExternal(centroidsFilepath, centroidsPerDim);
  } else if (mode == "simple_pca") {
    writeCentroidsExternal(centroidsFilepath, centroidsEigen);
  } else {
    writeCentroidsExternal(centroidsFilepath, km.last_centroids);
  }

  std::cout << "done" << std::endl;
  return 0;
}