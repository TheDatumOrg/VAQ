#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/sysinfo.h>
#include <vector>
#include <getopt.h>

#include "BitVecEngine.hpp"
#include "utils/Types.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/Experiment.hpp"
#include "utils/IO.hpp"

struct Args {
  std::string datasetFilepath = "";
  std::string queriesFilepath = "";
  std::string centroidsFilepath = "";
  int datasetSize = 0;
  int queriesSize = 0;
  int bitvecSize = 0;  // bitvec size
  int N = 1;  // dimension
  std::string datasetOriFilepath = "";
  std::string queriesOriFilepath = "";
  std::string groundtruthFilepath = "";
  std::string groundtruthFormat = "bin";
  std::string clusterIndexFilepath = "";
  int K = 100;
  int rerankFactor = 2;
  int thread = 20;
  int batchNum = 1e8; // 100 Million
  int queryMode = 0;
  int searchMethod = 1;
  bool writeClusterIndex = false;
  bool writeGroundtruth = false;
  int iteration = 1;
  std::string fileFormatOri = "fvecs";
  bool rescaleDataset = false;
  bool computeClassification = false;
} args;

int main(int argc, char **argv) {
  // parse arguments
  while (true) {
    static struct option long_options[] = {
      {"dataset", required_argument, 0, 'd'},
      {"queries", required_argument, 0, 'q'},
      {"dataset-ori", required_argument, 0, 's'},
      {"queries-ori", required_argument, 0, 'i'},
      {"bitvec-size", required_argument, 0, 'v'},
      {"timeseries-size", required_argument, 0, 't'},
      {"dataset-size", required_argument, 0, 'a'},
      {"queries-size", required_argument, 0, 'u'},
      {"k", required_argument, 0, 'k'},
      {"file-format-ori", required_argument, 0, 'z'},
      {"groundtruth", required_argument, 0, 'g'},
      {"groundtruth-format", required_argument, 0, 'y'},
      {"mode", required_argument, 0, 'm'},
      {"method", required_argument, 0, 'e'},
      {"centroids", required_argument, 0, 'c'},
      {"cluster-index", required_argument, 0, 'l'},
      {"write-cluster-index", required_argument, 0, 'r'},
      {"rerank-factor", required_argument, 0, 'f'},
      {"write-groundtruth", required_argument, 0, 'w'},
      {"batch", required_argument, 0, 'h'},
      {"iteration", required_argument, 0, 'o'},
      {"thread", required_argument, 0, 'b'},
      {"rescale", required_argument, 0, 'x'},
      {"compute-classification", required_argument, 0, 'j'},
      {"help", no_argument, 0, '?'}
    };
    int option_index = 0;

    int c = getopt_long (argc, argv, "", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
      case 'd': args.datasetFilepath = optarg; break;
      case 'q': args.queriesFilepath = optarg; break;
      case 's': args.datasetOriFilepath = optarg; break;
      case 'i': args.queriesOriFilepath = optarg; break;
      case 'v': args.bitvecSize = std::atoi(optarg); break;
      case 't': args.N = std::atoi(optarg); break;
      case 'a': args.datasetSize = std::atoi(optarg); break;
      case 'u': args.queriesSize = std::atoi(optarg); break;
      case 'k': args.K = std::atoi(optarg); break;
      case 'z': args.fileFormatOri = optarg; break;
      case 'g': args.groundtruthFilepath = optarg; break;
      case 'y': args.groundtruthFormat = optarg; break;
      case 'm': args.queryMode = std::atoi(optarg); break;
      case 'e': args.searchMethod = std::atoi(optarg); break;
      case 'c': args.centroidsFilepath = optarg; break;
      case 'l': args.clusterIndexFilepath = optarg; break;
      case 'r': args.writeClusterIndex = std::atoi(optarg); break;
      case 'f': args.rerankFactor = std::atoi(optarg); break;
      case 'b': args.thread = std::atoi(optarg); break;
      case 'w': args.writeGroundtruth = std::atoi(optarg); break;
      case 'h': args.batchNum = std::atoi(optarg); break;
      case 'o': args.iteration = std::atoi(optarg); break;
      case 'x': args.rescaleDataset = std::atoi(optarg); break;
      case 'j': args.computeClassification = std::atoi(optarg); break;
      case '?':
        std::cout << 
        "Usage:\n\
        \t--k XX \t\t\tK in K-nearest-neighbor\n\
        \t--dataset XX \t\t\tThe path to the binary dataset file\n\
        \t--queries XX \t\t\tThe path to the binary queries file\n\
        \t--dataset-ori XX \t\t\tThe path to the original dataset file\n\
        \t--queries-ori XX \t\t\tThe path to the original queries file\n\
        \t--dataset-size XX \t\tThe number of time series to load\n\
        \t--queries-size XX \t\tThe number of queries to run\n\
        \t--bitvec-size XX \t\tThe number bits in BitVector\n\
        \t--timeseries-size XX \t\tThe number of dimension\n\
        \t--iteration XX \t\tThe number of iteration\n\
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
        \t--mode: \n\
        \t--\t0=binary: only\n\
        \t--\t1=binary: rerank\n\
        \t--\t2=binary: use cluster info\n\
        \t--\t3=binary: use cluster info and rerank\n\
        \t--\t4=ED: only\n\
        \t--\t5=ED: use cluster info\n\
        \t--\t6=ED: with parallelization\n\
        \t--\t7=ED: with parallelization disk resident\n\
        \t--\t8=ED: use cluster info, parallelization, and disk resident\n\
        \t--method: \n\
        \t--\t0=Heap\n\
        \t--\t1=Sort\n\
        \t--\t2=Heap Early Abandon\n\
        \t--\t3=Sort Early Abandon\n\
        \t--centroids XX \t\tThe path to the binary centroids file\n\
        \t--cluster-index XX\t\tThe path to the cluster index file\n\
        \t--write-cluster-index\t\t\twrite out cluster index\n\
        \t--write-groundtruth\t\t\twrite out groundtruth\n\
        \t--rerank-factor XX\t\t\tThe factor when using rerank mode\n\
        \t--thread XX\t\t\tThread number if using parallel query\n\
        \t--rescale XX\t\t\tIf dataset will be rescaled to have variance >> 1\n\
        \t--compute-classification XX\t\t\tIf classification accuracy needs to be computed\n\
        \t--help\n\n\
        \t--**********************EXAMPLES**********************\n\n\
        \t--./main      --dataset XX --queries XX --k 100       \n\n\
        \t--            --mode 1 --method 1                     \n\n\
        \t--****************************************************\n\n" << std::endl;
        exit(0);
        break;
    }
  }

  std::cout << "\t==== Arguments info ====" << std::endl;
  switch (args.queryMode) {
    case 0: std::cout << "\tmode = binary: only" << std::endl; break;
    case 1: std::cout << "\tmode = binary: rerank" << std::endl; break;
    case 2: std::cout << "\tmode = binary: use cluster info" << std::endl; break;
    case 3: std::cout << "\tmode = binary: use cluster info and rerank" << std::endl; break;
    case 4: std::cout << "\tmode = ED: only" << std::endl; break;
    case 5: std::cout << "\tmode = ED: use cluster info" << std::endl; break;
    case 6: std::cout << "\tmode = ED: with parallelization" << std::endl; break;
    case 7: std::cout << "\tmode = ED: with parallelization disk resident" << std::endl; break;
    case 8: std::cout << "\tmode = ED: use cluster info, parallelization, and disk resident" << std::endl; break;
    case 9: std::cout << "\tmode = ED: with triangle inequality prune" << std::endl; break;
  }

  switch (args.searchMethod) {
    case 0: std::cout << "\tmethod = Heap" << std::endl; break;
    case 1: std::cout << "\tmethod = Sort" << std::endl; break;
    case 2: std::cout << "\tmethod = HeapEarlyAbandon" << std::endl; break;
    case 3: std::cout << "\tmethod = SortEarlyAbandon" << std::endl; break;
  }

  std::cout << "\tK: " << args.K << std::endl;
  std::cout << "\tdimension: " << args.N << std::endl;
  std::cout << "\tdataset size: " << args.datasetSize << std::endl;
  std::cout << "\tqueries size: " << args.queriesSize << std::endl;

  if (args.queryMode == 1 || (args.queryMode >= 3 && args.queryMode <= 9)) {
    if (args.fileFormatOri == "ascii") std::cout << "\toriginal file format = ascii" << std::endl;
    else if (args.fileFormatOri == "fvecs") std::cout << "\toriginal file format = fvecs" << std::endl;
    else if (args.fileFormatOri == "bvecs") std::cout << "\toriginal file format = bvecs" << std::endl;
    else if (args.fileFormatOri == "bin") std::cout << "\toriginal file format = bin" << std::endl;
  }
  std::cout << "\t==== ++++++++++++++ ====\n" << std::endl;
  
  bitvectors bvDataset, bvQueries;
  Matrixf oriDataset, oriQueries, centroids;
  Matrixi topnn;
  std::vector<int> clusterIdx;
  std::vector<int> datasetClassInfo, queryClassInfo;

  // read binary dataset and query
  if (args.queryMode >= 0 && args.queryMode <= 3) { 
    readFromExternal(args.datasetFilepath, bvDataset, args.bitvecSize, ',');
    readFromExternal(args.queriesFilepath, bvQueries, args.bitvecSize, ',');
  }
  // read original dataset
  if ((args.queryMode >= 1 && args.queryMode <= 6) || args.queryMode == 9) {
    if (args.fileFormatOri == "ascii") {
      readOriginalFromExternal<true>(args.datasetOriFilepath, oriDataset, args.N);
    } else if (args.fileFormatOri == "fvecs") {
      readFVecsFromExternal(args.datasetOriFilepath, oriDataset, args.N, args.datasetSize);
    } else if (args.fileFormatOri == "bvecs") {
      readBVecsFromExternal(args.datasetOriFilepath, oriDataset, args.N, args.datasetSize);
    } else if (args.fileFormatOri == "bin") {
      readFromExternalBin(args.datasetOriFilepath, oriDataset, args.N, args.datasetSize);
    }
  }
  // read original query
  if (args.queryMode >= 1 && args.queryMode <= 9) { 
    if (args.fileFormatOri == "ascii") {
      readOriginalFromExternal<true>(args.queriesOriFilepath, oriQueries, args.N);
    } else if (args.fileFormatOri == "fvecs") {
      readFVecsFromExternal(args.queriesOriFilepath, oriQueries, args.N, args.queriesSize);
    } else if (args.fileFormatOri == "bvecs") {
      readBVecsFromExternal(args.queriesOriFilepath, oriQueries, args.N, args.queriesSize);
    } else if (args.fileFormatOri == "bin") {
      readFromExternalBin(args.queriesOriFilepath, oriQueries, args.N, args.queriesSize);
    }
  }
  if (args.rescaleDataset) {
    float scaleMean = 0;
    std::vector<float> means(args.N, 0);
    std::vector<float> variances(args.N, 0);
    for (int j=0; j<args.N; j++) {
      for (int i=0; i<args.datasetSize; i++) {
        means[j] += oriDataset[i][j];
      }
      means[j] /= args.datasetSize;

      for (int i=0; i<args.datasetSize; i++) {
        variances[j] += (oriDataset[i][j] - means[j]) * (oriDataset[i][j] - means[j]);
      }
      variances[j] /= (args.datasetSize-1);

      scaleMean += sqrtf(1.0 / variances[j]);
    }
    scaleMean /= args.N;

    std::cout << "Scale mean = " << scaleMean << std::endl;
    for (auto &row: oriDataset) {
      for (auto &col: row) {
        col *= scaleMean;
      }
    }
    for (auto &row: oriQueries) {
      for (auto &col: row) {
        col *= scaleMean;
      }
    }
  }
  // read cluster info
  if (args.queryMode == 2 || args.queryMode == 3 || args.queryMode == 5 || args.queryMode == 9) {
    readOriginalFromExternal(args.centroidsFilepath, centroids, args.N);
    if (args.queryMode != 9) {
      if (!args.writeClusterIndex) {
        readClusterIndexExternal(args.clusterIndexFilepath, clusterIdx);
      } else {
        computeClusterIndex(oriDataset, centroids, clusterIdx);
        writeClusterIndexExternal(args.clusterIndexFilepath, clusterIdx);
      }
    }
  }
  // read ground truth data
  if (!args.writeGroundtruth) {
    if (args.groundtruthFormat == "ascii") {
      readTOPNNExternal(args.groundtruthFilepath, topnn, args.K);
    } else if (args.groundtruthFormat == "ivecs") {
      readIVecsFromExternal(args.groundtruthFilepath, topnn, args.K);
    } else if (args.groundtruthFormat == "bin") {
      readTOPNNExternalBin(args.groundtruthFilepath, topnn, args.K);
    }
  }
  // read classification info (still support only ascii file format)
  if (args.computeClassification) {
    readClassificationInfoFromExternal(args.datasetOriFilepath, datasetClassInfo);
    readClassificationInfoFromExternal(args.queriesOriFilepath, queryClassInfo);
  }

  std::vector<std::vector<IdxDistPair>> pairs;
  std::vector<std::vector<IdxDistPairFloat>> pairsFloat;

  BitVecEngine engine(args.bitvecSize);
  if (args.queryMode == 0) {
    engine.loadBitV(bvDataset);
  } else if (args.queryMode == 1) {
    engine.loadBitV(bvDataset);
    engine.loadOriginal(oriDataset);
  } else if (args.queryMode == 2 || args.queryMode == 3) {
    engine.loadBitV(bvDataset);
    engine.loadOriginal(oriDataset);
    engine.loadCentroids(centroids);
    engine.loadClusterInfo(clusterIdx, centroids.size());
  } else if (args.queryMode == 4 || args.queryMode == 6) { 
    engine.loadOriginal(oriDataset);
  } else if (args.queryMode == 5) {
    engine.loadOriginal(oriDataset);
    engine.loadCentroids(centroids);
    engine.loadClusterInfo(clusterIdx, centroids.size());
  } else if (args.queryMode == 9) {
    engine.loadOriginal(oriDataset);
    engine.loadCentroids(centroids);
    engine.computeTriangleInequalityClusters();
  }
  
  double avgElapsed = 0;
  for (int repIter=0; repIter<args.iteration; repIter++) {
    std::cout << "Iteration " << repIter << " elapsed : ";

    cputime_t startTime = timeNow(), endTime;
    switch (args.queryMode) {
      case 0: pairs = engine.query(bvQueries, args.K, args.searchMethod); break;
      case 1: pairsFloat = engine.queryRerank(bvQueries, oriQueries, args.K, args.rerankFactor, args.searchMethod); break;
      case 2: pairs = engine.queryWithClusterInfo(bvQueries, oriQueries, args.K, args.searchMethod, 1); break;
      case 3: pairsFloat = engine.queryRerankWithClusterInfo(bvQueries, oriQueries, args.K, args.rerankFactor, args.searchMethod, 1); break;
      case 4: pairsFloat = engine.queryNaive(oriQueries, args.K, args.searchMethod); break;
      case 5: pairsFloat = engine.queryNaiveWithClusterInfo(oriQueries, args.K, args.searchMethod, 1); break;
      case 6: pairsFloat = engine.queryNaiveParallel(oriQueries, args.K, args.thread); break;
      case 7: pairsFloat = engine.queryNaiveParallelDiskResident(args.datasetOriFilepath, oriQueries, args.K, args.thread, args.searchMethod, args.batchNum); break;
      case 8: pairsFloat = engine.queryNaiveWithClusterInfoParallelDiskResident(args.datasetOriFilepath, oriQueries, clusterIdx, args.K, args.thread, args.searchMethod, args.batchNum, 1); break;
      case 9: pairsFloat = engine.queryNaiveTriangleInequality(oriQueries, args.K, args.searchMethod); break;
    }
    endTime = timeNow();
    
    double elapsed = durationS(startTime, endTime);
    std::cout << elapsed << " seconds" << std::endl;
    avgElapsed += elapsed;
  }
  avgElapsed /= args.iteration;
  
  if (args.writeGroundtruth) {
    if (args.queryMode == 0 || args.queryMode == 2) {
      int rowIter = 0;
      for (auto row: pairs) {
        topnn.push_back(std::vector<int>());
        for (auto col: row) {
          topnn[rowIter].push_back(col.idx);
        }
        rowIter++;
      }
    } else {
      int rowIter = 0;
      for (auto row: pairsFloat) {
        topnn.push_back(std::vector<int>());
        for (auto col: row) {
          topnn[rowIter].push_back(col.idx);
        }
        rowIter++;
      }
    }
    writeTOPNNExternalBin(args.groundtruthFilepath, topnn);
  } else {
    std::cout << "Quality Measurement" << std::endl;
    if (args.computeClassification) {
      std::vector<int> nnList = {1, 5, 10};
      for (auto nn: nnList) {
        std::vector<int> matchedClass(3, 0);
        for (int p_idx=0; p_idx<(int)pairs.size(); p_idx++) {
          std::vector<int> matchedCount(3, 0);
          for (int i=0; i<nn; i++) {
            int idx = pairs[p_idx][i].idx;
            matchedCount[datasetClassInfo[idx]-1] += 1;
          }

          if (matchedCount[0] >= matchedCount[1] && matchedCount[0] >= matchedCount[2]) {
            matchedClass[0] += (queryClassInfo[p_idx] == 1) ? 1 : 0;
          } else if (matchedCount[1] >= matchedCount[0] && matchedCount[1] >= matchedCount[2]) {
            matchedClass[1] += (queryClassInfo[p_idx] == 2) ? 1 : 0;
          } else {
            matchedClass[2] += (queryClassInfo[p_idx] == 3) ? 1 : 0;
          }
        }

        std::cout << "classification accuracy with " << nn << "NN: " << ((float)(matchedClass[0]+matchedClass[1]+matchedClass[2]) / topnn.size()) << std::endl;
      }
    } else {
      if (args.queryMode == 0 || args.queryMode == 2) {
        std::cout << "\tprecision(avg_recall): " << getAvgRecall(pairs, topnn, args.K) << std::endl;
        std::cout << "\trecall@R: " << getRecallAtR(pairs, topnn) << std::endl;
        std::cout << "\tMAP: " << getMeanAveragePrecision(pairs, topnn, args.K) << std::endl;
      } else {
        std::cout << "\tprecision(avg_recall): " << getAvgRecall(pairsFloat, topnn, args.K) << std::endl;
        std::cout << "\trecall@R: " << getRecallAtR(pairsFloat, topnn) << std::endl;
        std::cout << "\tMAP: " << getMeanAveragePrecision(pairsFloat, topnn, args.K) << std::endl;
      }
    }
  }
  
  std::cout << "Average time: " << avgElapsed << " seconds\n" << std::endl;

  return 0;
}