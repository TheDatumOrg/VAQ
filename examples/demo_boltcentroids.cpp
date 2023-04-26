#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "KMeans.hpp"
#include "utils/Experiment.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/IO.hpp"
#include "utils/Types.hpp"

int main(int argc, char **argv) {
  std::vector<ArgsParse::opt> long_options{
    {"dataset", 's', ""},
    {"file-format-ori", 's', "fvecs"},
    {"save", 's', ""},
    {"timeseries-size", 'i', "1"},
    {"dataset-size", 'i', "0"},
    {"subspace", 'i', "32"},
    {"bitbudget", 'i', "256"},
  };
  ArgsParse args = ArgsParse(argc, argv, long_options, "HELP");
  args.printArgs();

  if (!isFileExists(args["dataset"])) {
    std::cerr << "Dataset file doesn't exists" << std::endl;
    return 1;
  }

  int dimPadding = 0;
  if (args.at<int>("timeseries-size") % args.at<int>("subspace") != 0) {
    int subvectorlen = (int) std::ceil((float)args.at<int>("timeseries-size") / args.at<int>("subspace"));
    dimPadding = (subvectorlen * args.at<int>("subspace")) - args.at<int>("timeseries-size");
  }
  Eigen::MatrixXf dataset = Eigen::MatrixXf::Zero(args.at<int>("dataset-size"), args.at<int>("timeseries-size") + dimPadding);
  if (args["file-format-ori"] == "ascii") {
    readOriginalFromExternal<true>(args["dataset"], dataset, args.at<int>("timeseries-size"), ',');
  } else if (args["file-format-ori"] == "fvecs") {
    readFVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
  } else if (args["file-format-ori"] == "bvecs") {
    readBVecsFromExternal(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
  } else if (args["file-format-ori"] == "bin") {
    readFromExternalBin(args["dataset"], dataset, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
  }

  START_TIMING(PQ);
  CentroidsPerSubsType centroidsPerDim(args.at<int>("subspace"));
  const int subVectorLength = dataset.cols() / args.at<int>("subspace");
  for (int i=0; i<args.at<int>("subspace"); i++) {
    std::cout << "Training slice " << (i+1) << "/" << args.at<int>("subspace") << std::endl;
    std::cout << "Clustering " << dataset.rows() << " points in " << subVectorLength << "D to " << (1 << 4) << " clusters, 25 iterations" << std::endl;
    START_TIMING(PQ_SLICE);
    centroidsPerDim[i] = KMeans::staticFitSampling(
      dataset.block(0, i * subVectorLength, dataset.rows(), subVectorLength),
      1 << 4,
      25,
      1 << 4,
      true  // verbose
    );
    END_TIMING(PQ_SLICE, "  Training slice: ");
  }
  END_TIMING(PQ, "PQ Training time: ");

  // writeout results for bolt
  writeCentroidsExternalBolt(args["save"], centroidsPerDim);

  return 0;
}