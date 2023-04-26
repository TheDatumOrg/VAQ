#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <getopt.h>

#include <sys/sysinfo.h>
#include <sys/stat.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "VAQ.hpp"
#include "BitVecEngine.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/Experiment.hpp"
#include "utils/IO.hpp"

int main(int argc, char **argv) {
  std::vector<ArgsParse::opt> long_options {
    {"dataset", 's', ""},
    {"queries", 's', ""},
    {"file-format-ori", 's', "fvecs"},
    {"groundtruth", 's', ""},
    {"groundtruth-format", 's', "ascii"},
    {"result", 's', ""},
    {"timeseries-size", 'i', "1"},
    {"dataset-size", 'i', "0"},
    {"queries-size", 'i', "0"},
    {"k", 'i', "100"},
    {"bitbudget", 'i', "256"},
    {"sampling-ratio", 'f', "0.1"}
  };
  ArgsParse args = ArgsParse(argc, argv, long_options, "HELP");
  args.printArgs();

  // check if dataset and queries exist
  if (!isFileExists(args["dataset"]) || !isFileExists(args["queries"])) {
    std::cerr << "Dataset or queries file doesn't exists" << std::endl;
    return 1;
  }

  RowMatrixXf dataset = RowMatrixXf::Zero(args.at<int>("dataset-size"), args.at<int>("timeseries-size"));
  RowMatrixXf traindataset, encodedataset;
  RowMatrixXf queries = RowMatrixXf::Zero(args.at<int>("queries-size"), args.at<int>("timeseries-size"));
  std::vector<std::vector<int>> topnn;

  {
    std::cout << "Read dataset" << std::endl;
    if (args["file-format-ori"] == "ascii") {
      readOriginalFromExternal<true>(args["dataset"], dataset, args.at<int>("timeseries-size"), ',');
    } else if (args["file-format-ori"] == "fvecs") {
      readFVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
    } else if (args["file-format-ori"] == "bvecs") {
      readBVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
    } else if (args["file-format-ori"] == "bin") {
      readFromExternalBin(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
    }

    std::cout << "Read queries" << std::endl;
    if (args["file-format-ori"] == "ascii") {
      readOriginalFromExternal<true>(args["queries"], queries, args.at<int>("timeseries-size"), ',');
    } else if (args["file-format-ori"] == "fvecs") {
      readFVecsFromExternal(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
    } else if (args["file-format-ori"] == "bvecs") {
      readBVecsFromExternal(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
    } else if (args["file-format-ori"] == "bin") {
      readFromExternalBin(args["queries"], queries, args.at<int>("timeseries-size"), args.at<int>("queries-size"));
    }

    std::cout << "Read groundtruth" << std::endl;
    if (args["groundtruth"] != "") {
      std::cout << "Read groundtruth" << std::endl;
      if (args["groundtruth-format"] == "ascii") {
        readTOPNNExternal(args["groundtruth"], topnn, 100, ',');
      } else if (args["groundtruth-format"] == "ivecs") {
        readIVecsFromExternal(args["groundtruth"], topnn, args.at<int>("k"));
      } else if (args["groundtruth-format"] == "bin") {
        readTOPNNExternalBin(args["groundtruth"], topnn, args.at<int>("k"));
      }
    }
  }
  std::cout << "Training & encoding phase" << std::endl;
  // constexpr int max_points_per_d = 1000;
  bool useSampling = false;
  // int sampleSize = max_points_per_d * dataset.cols();
  int sampleSize = 10000000;
  if (dataset.rows() > sampleSize) {
    useSampling = true;
    std::vector<int> perm(dataset.rows());
    randomPermutation(perm);
    
    traindataset.resize(sampleSize, dataset.cols());
    for (int i=0; i<sampleSize; i++) {
      traindataset.row(i) = dataset.row(perm[i]);
    }
  }
  
  bool useSamplingEnc = false;
  if (dataset.rows() > 10000000) {
    int sampleSizeEnc = static_cast<int>(args.at<float>("sampling-ratio") * (float)dataset.rows());
    useSamplingEnc = true;
    std::vector<int> perm(dataset.rows());
    randomPermutation(perm);
    
    encodedataset.resize(sampleSizeEnc, dataset.cols());
    if (args["groundtruth"] != "") {
      std::vector<int> sampledIdx;
      sampledIdx.reserve(sampleSizeEnc);
      for (int row=0; row<(int)topnn.size(); row++) {
        for (int col=0; col<args.at<int>("k"); col++) {
          int val = topnn[row][col];
          if (!std::binary_search(sampledIdx.begin(), sampledIdx.end(), val)) {
            // insert sorted
            sampledIdx.insert(
              std::upper_bound(sampledIdx.begin(), sampledIdx.end(), val),
              val
            );
          }
        }
      }
      for (int i=0; i<sampleSizeEnc; i++) {
        encodedataset.row(i) = dataset.row(sampledIdx[i]);
      }
      int currSampled = sampledIdx.size();
      int i=0;
      while ((currSampled < sampleSizeEnc) && (i < sampleSizeEnc)) {
        if (!std::binary_search(sampledIdx.begin(), sampledIdx.end(), perm[i]))  {
          encodedataset.row(i) = dataset.row(perm[i]);
          currSampled += 1;
        }
        i += 1;
      }
      assert(currSampled == sampleSizeEnc);
    } else {
      for (int i=0; i<sampleSizeEnc; i++) {
        encodedataset.row(i) = dataset.row(perm[i]);
      }
    }

    // create new groundtruth
    BitVecEngine engine(0);
    std::vector<std::vector<IdxDistPairFloat>> pairsFloat;
    pairsFloat = engine.queryNaiveEigen(encodedataset, queries, args.at<int>("k"));
    if (args["groundtruth"] == "") {
      topnn.resize(queries.rows(), std::vector<int>(args.at<int>("k")));
    }
    for (int row=0; row<(int)pairsFloat.size(); row++) {
      for (int col=0; col<(int)pairsFloat[row].size(); col++)  {
        topnn[row][col] = pairsFloat[row][col].idx;
      }
    }
  }

  // table consist of cols: [ m, min, max, precision, query time ]
  RowMatrix<double> precisionTable(62, 5);
  int resultCounter = 0;
  double bestPrecision = 0;

  int counter = 1;
  cputime_t paramSearchStart;

  auto evaluateParams = [&args, &topnn, &bestPrecision, &precisionTable, useSampling, useSamplingEnc, &dataset, &traindataset, &encodedataset, &queries, &paramSearchStart, &counter, &resultCounter](
    const int m, const int minBits, const int maxBits) {
    VAQ vaq;
    vaq.mBitBudget = args.at<int>("bitbudget");
    vaq.mSubspaceNum = m;
    vaq.mPercentVarExplained = 1.0f;
    vaq.mMinBitsPerSubs = minBits;
    vaq.mMaxBitsPerSubs = maxBits;
    vaq.mMethods = VAQ::NNMethod::EA;

    double currDuration = durationS(paramSearchStart, timeNow());
    if (std::ceil(currDuration) / 60 >= counter) {
      counter = ((int)std::ceil(currDuration)) / 60 + 1;
      std::cout << "[" << currDuration << " s] - ";
      std::cout << "Params Tested: " << resultCounter << std::endl;
      std::cout << "Traning VAQ" << vaq.mBitBudget << "m" << m << "min" << minBits << "max" << maxBits << "var" << 1 << ",EA" << std::endl;
    }

    // encoding phase
    if (!useSampling) {
      vaq.train(dataset, false);
    } else {
      vaq.train(traindataset, false);
    }

    if (!useSamplingEnc) {
      vaq.encode(dataset);
    } else {
      vaq.encode(encodedataset);
    }

    cputime_t start = timeNow();
    std::vector<std::vector<IdxDistPairFloat>> answers = vaq.search(queries, args.at<int>("k"));
    cputime_t end = timeNow();
    double duration = durationS(start, end);
    double precision = getAvgRecall(answers, topnn, args.at<int>("k"));
    if (precision > bestPrecision) { 
      bestPrecision = precision;
      std::cout << "new best Precision: " << bestPrecision << std::endl;
    }
    precisionTable(resultCounter, 0) = m;
    precisionTable(resultCounter, 1) = minBits;
    precisionTable(resultCounter, 2) = maxBits;
    precisionTable(resultCounter, 3) = precision;
    precisionTable(resultCounter, 4) = duration;
    resultCounter += 1;
  };

  paramSearchStart = timeNow();
  for (const int& m: { 32}) {
  // for (const int& m: { 32, 64 }) {
    if (m == 32) {
      for (const int& minBits: { 1, 2, 3, 4, 5, 6, 7 }) {
      // for (const int& minBits: { 4 }) {
        for (const int& maxBits: { 9, 10, 11, 12, 13 }) {
        // for (const int& maxBits: { 10, 11 }) {
          evaluateParams(m, minBits, maxBits);
        }
      }
    } else if (m == 64){
      for (const int& minBits: { 1, 2, 3 }) {
        for (const int& maxBits: { 5, 6, 7, 8, 9, 10, 11, 12, 13 }) {
          evaluateParams(m, minBits, maxBits);
        }
      }
    }
  }

  double bestPrecisionFinal = 0;
  int bestPrecisionIndex = 0;
  for (int i=0; i<resultCounter; i++) {
    if (precisionTable(i, 3) > bestPrecisionFinal) {
      bestPrecisionIndex = i;
      bestPrecisionFinal = precisionTable(i, 3);
    }
  }
  std::cout << "Best parameters: " << std::endl;
  std::cout << "subvector: " << precisionTable(bestPrecisionIndex, 0) << std::endl;
  std::cout << "min bits: " << precisionTable(bestPrecisionIndex, 1) << std::endl;
  std::cout << "max bits: " << precisionTable(bestPrecisionIndex, 2) << std::endl;
  std::cout << "with precision = " << bestPrecisionFinal << std::endl;

  if (args["result"] != "") {
    std::cout << "Writing paramsearch results" << std::endl;
    {
      std::ofstream outfile;
      outfile.open(args["result"]);
      outfile << "m,minbits,maxbits,precision" << std::endl;
      for (int i=0; i<resultCounter; i++) {
        outfile << precisionTable(i, 0) << ',';
        outfile << precisionTable(i, 1) << ',';
        outfile << precisionTable(i, 2) << ',';
        outfile << precisionTable(i, 3) << std::endl;
      }
      outfile.close();
    }
  }

  return 0;
}
