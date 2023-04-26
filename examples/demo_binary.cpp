#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/sysinfo.h>
#include <vector>
#include <getopt.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "BitVecEngine.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/IO.hpp"
#include "utils/Experiment.hpp"

using Matrixf = std::vector<std::vector<float>>;

struct Args {
  std::string datasetFilepath = "";
  std::string queriesFilepath = "";
  int datasetSize = 0;
  int queriesSize = 0;
  int bitvecSize = 0;  // bitvec size
  int N = 1;  // dimension
  std::string datasetOriFilepath = "";
  std::string queriesOriFilepath = "";
  std::string groundtruthFilepath = "";
  std::string groundtruthFormat = "ascii";
  std::string fileFormatOri = "fvecs";
  bool isLUT = false;
  bool isVAQ = false;
  int M = 32; // subvector length
  float variance = 1; // compression
  int minbits = 4; // min bits per segment
  int maxbits = 8; // max bits per segment
  int K = 100;
  int clusterNum = 100;
  int pruneComp = 1;
  uint32_t method = BitVecEngine::NNMethod::Sort;
} args;

int main(int argc, char **argv) {
  while (true) {
    static struct option long_options[] = {
      {"dataset", required_argument, 0, 'd'},
      {"queries", required_argument, 0, 'q'},
      {"dataset-ori", required_argument, 0, 's'},
      {"queries-ori", required_argument, 0, 'i'},
      {"groundtruth", required_argument, 0, 'g'},
      {"groundtruth-format", required_argument, 0, 'y'},
      {"bitvec-size", required_argument, 0, 'v'},
      {"timeseries-size", required_argument, 0, 't'},
      {"dataset-size", required_argument, 0, 'a'},
      {"queries-size", required_argument, 0, 'u'},
      {"file-format-ori", required_argument, 0, 'z'},
      {"lut", required_argument, 0, 'l'},
      {"vaq", required_argument, 0, 'b'},
      {"m", required_argument, 0, 'm'},
      {"var", required_argument, 0, 'r'},
      {"minbits", required_argument, 0, 'n'},
      {"maxbits", required_argument, 0, 'x'},
      {"method", required_argument, 0, 'o'},
      {"cluster-num", required_argument, 0, 'c'},
      {"prune-comp", required_argument, 0, 'e'},
      {"k", required_argument, 0, 'k'},
      {"help", no_argument, 0, '?'}
    };

    int option_index = 0;
    int raw_method;

    int c = getopt_long (argc, argv, "", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
      case 'd': args.datasetFilepath = optarg; break;
      case 'q': args.queriesFilepath = optarg; break;
      case 's': args.datasetOriFilepath = optarg; break;
      case 'i': args.queriesOriFilepath = optarg; break;
      case 'g': args.groundtruthFilepath = optarg; break;
      case 'y': args.groundtruthFormat = optarg; break;
      case 'v': args.bitvecSize = std::atoi(optarg); break;
      case 't': args.N = std::atoi(optarg); break;
      case 'a': args.datasetSize = std::atoi(optarg); break;
      case 'u': args.queriesSize = std::atoi(optarg); break;
      case 'z': args.fileFormatOri = optarg; break;
      case 'l': args.isLUT = std::atoi(optarg); break;
      case 'b': args.isVAQ = std::atoi(optarg); break;
      case 'o': 
        args.method = 0u;
        raw_method = std::atoi(optarg);
        switch (raw_method) {
          case 0: args.method = BitVecEngine::NNMethod::Sort; break;
          case 1: args.method = BitVecEngine::NNMethod::EA; break;
          case 2: args.method = BitVecEngine::NNMethod::TI; break;
          case 3: args.method = BitVecEngine::NNMethod::EA | BitVecEngine::NNMethod::TI; break;
        }
        break;
      case 'm': args.M = std::atoi(optarg); break;
      case 'r': args.variance = std::atof(optarg); break;
      case 'n': args.minbits = std::atoi(optarg); break;
      case 'x': args.maxbits = std::atoi(optarg); break;
      case 'c': args.clusterNum = std::atoi(optarg); break;
      case 'e': args.pruneComp = std::atoi(optarg); break;
      case 'k': args.K = std::atoi(optarg); break;
      case '?':
        std::cout << 
        "Usage:\n\
        \t--dataset XX \t\t\tThe path to the binary dataset file\n\
        \t--queries XX \t\t\tThe path to the binary queries file\n\
        \t--dataset-ori XX \t\t\tThe path to the original dataset file\n\
        \t--queries-ori XX \t\t\tThe path to the original queries file\n\
        \t--dataset-size XX \t\tThe number of time series to load\n\
        \t--queries-size XX \t\tThe number of queries to run\n\
        \t--bitvec-size XX \t\tThe number bits in BitVector\n\
        \t--timeseries-size XX \t\tThe number of dimension\n\
        \t--file-format-ori: \n\
        \t--\tascii\n\
        \t--\tfvecs\n\
        \t--\tbvecs\n\
        \t--\tbin\n\
        \t--groundtruth XX \t\tThe path to the groundtruth file\n\
        \t--groundtruth-format: \n\
        \t--\tascii\n\
        \t--\tbin\n\
        \t--\tivecs\n\
        \t--method: \n\
        \t--\t0: Sort\n\
        \t--\t1: Early abandon\n\
        \t--\t2: Triangle Inequality Pruning\n\
        \t--\t3: EA + TI\n\
        \t--lut\t\tQuery KNN using BE and LUT\n\
        \t--vaq\t\tQuery KNN using Variance Aware Quantization\n\
        \t--m\t\tNumber of SubVector\n\
        \t--k XX\t\tK in K-nearest-neighbor\n\
        \t--var XX\t\tvariance explained\n\
        \t--minbits XX\t\tMinBits\n\
        \t--maxbits XX\t\tMaxBits\n\
        \t--cluster-num XX\t\tNumber of cluster to help pruning\n\
        \t--prune-comp XX\t\tFirst x component to use as reduce table\n\
        \t--help\n\n\
        \t--**********************EXAMPLES**********************\n\n\
        \t--./mainencode --dataset XX --queries XX              \n\n\
        \t--             --file-format-ori bin                  \n\n\
        \t--****************************************************\n\n" << std::endl;
        exit(0);
        break;
    }

  }
  
  std::cout << "\t==== ++++++++++++++ ====\n" << std::endl;
  std::cout << "\tvariance: " << args.variance << std::endl;
  std::cout << "\tdimension: " << args.N << std::endl;
  std::cout << "\tbitbudget: " << args.bitvecSize << std::endl;
  std::cout << "\tsubvector: " << args.M << std::endl;
  std::cout << "\tminbits: " << args.minbits << std::endl;
  std::cout << "\tmaxbits: " << args.maxbits << std::endl;
  std::cout << "\tdataset size: " << args.datasetSize << std::endl;
  std::cout << "\tqueries size: " << args.queriesSize << std::endl;

  if (args.fileFormatOri == "ascii") std::cout << "\toriginal file format = ascii" << std::endl;
  else if (args.fileFormatOri == "fvecs") std::cout << "\toriginal file format = fvecs" << std::endl;
  else if (args.fileFormatOri == "bvecs") std::cout << "\toriginal file format = bvecs" << std::endl;
  else if (args.fileFormatOri == "bin") std::cout << "\toriginal file format = bin" << std::endl;
  std::cout << "\t==== ++++++++++++++ ====\n" << std::endl;

  // make sure vector length / sub vector is even
  int nAppendColumn = 0;
  if ((float)args.N / args.M != args.N / args.M) {
    nAppendColumn = (std::ceil((float)args.N / args.M) * args.M) - args.N;
  }

  std::vector<std::vector<uint64_t>> bvDataset(args.datasetSize, std::vector<uint64_t>(actualBitVLen(args.bitvecSize), 0)),
                                     bvQueries(args.queriesSize, std::vector<uint64_t>(actualBitVLen(args.bitvecSize), 0));
  Eigen::MatrixXf oriDataset(args.datasetSize, args.N + nAppendColumn), oriQueries(args.queriesSize, args.N + nAppendColumn);
  // Eigen::MatrixXf oriDataset = Eigen::MatrixXf::Random(args.datasetSize, args.N + nAppendColumn);
  // Eigen::MatrixXf oriQueries = Eigen::MatrixXf::Random(args.queriesSize, args.N + nAppendColumn);
  Matrixi topnn;

  // read original dataset
  if (args.fileFormatOri == "ascii") {
    readOriginalFromExternal<true>(args.datasetOriFilepath, oriDataset, args.N, ',');
    readOriginalFromExternal<true>(args.queriesOriFilepath, oriQueries, args.N, ',');
  } else if (args.fileFormatOri == "fvecs") {
    readFVecsFromExternal(args.datasetOriFilepath, oriDataset, args.N, args.datasetSize);
    readFVecsFromExternal(args.queriesOriFilepath, oriQueries, args.N, args.queriesSize);
  } else if (args.fileFormatOri == "bvecs") {
    readBVecsFromExternal(args.datasetOriFilepath, oriDataset, args.N, args.datasetSize);
    readBVecsFromExternal(args.queriesOriFilepath, oriQueries, args.N, args.queriesSize);
  } else if (args.fileFormatOri == "bin") {
    readFromExternalBin(args.datasetOriFilepath, oriDataset, args.N, args.datasetSize);
    readFromExternalBin(args.queriesOriFilepath, oriQueries, args.N, args.queriesSize);
  }

  // read ground truth data
  if (args.isLUT || args.isVAQ) {
    if (args.groundtruthFormat == "ascii") {
      readTOPNNExternal(args.groundtruthFilepath, topnn, 100, '\t');
    } else if (args.groundtruthFormat == "ivecs") {
      readIVecsFromExternal(args.groundtruthFilepath, topnn, args.K);
    } else if (args.groundtruthFormat == "bin") {
      readTOPNNExternalBin(args.groundtruthFilepath, topnn, args.K);
    }
  }
  
  if (args.isLUT) {
    BitVecEngine engine(args.bitvecSize);

    // encoding phase
    cputime_t startTime = timeNow(), endTime;
    CodebookType codebook;
    engine.binaryEncodingLUT(oriDataset, args.bitvecSize, codebook, true);
    endTime = timeNow();
    double elapsed = durationS(startTime, endTime);
    std::cout << "Encoding time: " << elapsed << std::endl;
    
    // query phase
    startTime = timeNow();
    std::vector<std::vector<IdxDistPairFloat>> pairs = engine.queryLUT(oriQueries, args.K, codebook);
    endTime = timeNow();
    elapsed = durationS(startTime, endTime);
    std::cout << "Query time: " << elapsed << std::endl;

    // std::cout << "Pairs " << pairs.size() << " " << pairs[0].size() << std::endl;
    // for (auto &a: pairs) {
    //   for (auto &b: a) {
    //     std::cout << "(" << b.idx << ", " << b.dist << "), ";
    //   }
    //   std::cout << std::endl;
    // }
    // exit(0);
    // std::cout << "TOPNN " << topnn.size() << " " << topnn[0].size() << std::endl;
    
    // Measure precision
    std::cout << "\tprecision(avg_recall): " << getAvgRecall<0>(pairs, topnn, args.K) << std::endl;
    std::cout << "\trecall@R: " << getRecallAtR<0>(pairs, topnn) << std::endl;
    std::cout << "\tMAP: " << getMeanAveragePrecision<0>(pairs, topnn, args.K) << std::endl;
  } else {  // encoding only
    // encoding phase
    cputime_t startTime = timeNow(), endTime;
    BitVecEngine::binaryEncoding(oriDataset, oriQueries, bvDataset, bvQueries, args.bitvecSize);
    // BitVecEngine::binaryEncodingSimple(oriDataset, oriQueries, bvDataset, bvQueries);
    endTime = timeNow();
    double elapsed = durationS(startTime, endTime);
    std::cout << "Encoding time: " << elapsed << std::endl;

    writeToExternal(args.datasetFilepath, bvDataset, args.bitvecSize);
    writeToExternal(args.queriesFilepath, bvQueries, args.bitvecSize);
  }
  std::cout << std::endl;

  return 0;
}