#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <getopt.h>

#include <sys/sysinfo.h>
#include <sys/stat.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "VAQ.hpp"
#include "utils/TimingUtils.hpp"
#include "utils/Experiment.hpp"
#include "utils/IO.hpp"

int main(int argc, char **argv) {
  std::vector<ArgsParse::opt> long_options {
    {"dataset", 's', ""},
    {"queries", 's', ""},
    {"file-format-ori", 's', "fvecs"},
    {"save", 's', ""},
    {"save-enc", 's', ""},
    {"groundtruth", 's', ""},
    {"groundtruth-format", 's', "ascii"},
    {"result", 's', ""},
    {"timeseries-size", 'i', "1"},
    {"dataset-size", 'i', "0"},
    {"queries-size", 'i', "0"},
    {"k", 'i', "100"},
    {"method", 's', "VAQ256m32min7max13var1,EA"},
    // VAQ256m32min7max13var1,SORT means VAQ 256 bits (32 bytes), 32 subvector, 7 min bits per segment, 1 variance (no compression), SORT the result
    // another description example: 
    // - VAQ64m16min3max6var0.99,EA
    // - PCA,PQ256m32,TI100
    // - VAQ128m32min6max9var0.95,EA_TI200
    // - VAQ128m32min6max9var0.95,Heap
    {"refine", 's', ""},
    {"hc-bitalloc", 's', ""},
    {"learn-ratio", 'f', "0.05"},
    {"visit-cluster", 'f', "1"},
    {"kmeans-ver", 'i', "0"}
  };
  ArgsParse args = ArgsParse(argc, argv, long_options, "HELP");
  args.printArgs();

  // check if dataset and queries exist
  if (!isFileExists(args["dataset"]) || !isFileExists(args["queries"])) {
    std::cerr << "Dataset or queries file doesn't exists" << std::endl;
    return 1;
  }

  VAQ vaq;
  vaq.parseMethodString(args["method"]);
  vaq.mVisit = args.at<float>("visit-cluster");
  if (args.at<int>("kmeans-ver") == 1) {
    vaq.mHierarchicalKmeans = true;
  } else if (args.at<int>("kmeans-ver") == 2) {
    vaq.mBinaryKmeans = true;
  }

  std::cout << "Preprocessing steps..\n" << std::endl;
  
  int dimPadding = 0;
  if (args.at<int>("timeseries-size") % vaq.mSubspaceNum != 0) {
    int subvectorlen = args.at<int>("timeseries-size") / vaq.mSubspaceNum;
    subvectorlen += (args.at<int>("timeseries-size") % vaq.mSubspaceNum > 0) ? 1 : 0;
    dimPadding = (subvectorlen * vaq.mSubspaceNum) - args.at<int>("timeseries-size");
  }
  RowMatrixXf dataset = RowMatrixXf::Zero(args.at<int>("dataset-size"), args.at<int>("timeseries-size") + dimPadding);
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

    std::cout << "Training & encoding phase" << std::endl;
    if (args["save"] != "" && isFileExists(args["save"])) {
      std::cout << "Reading saved centroids from " << args["save"] << std::endl;
      vaq.mCentroidsPerSubs = loadCentroids(args["save"]);
      vaq.mCentroidsPerSubsCMajor.resize(vaq.mCentroidsPerSubs.size());
      for (int i=0; i<(int)vaq.mCentroidsPerSubs.size(); i++) {
        vaq.mCentroidsPerSubsCMajor[i] = vaq.mCentroidsPerSubs[i];
      }
    }
    if (args["hc-bitalloc"] != "") {
      vaq.mBitsAlloc = parseVAQHardcode(args["hc-bitalloc"]);
      std::cout << "Hardcoded bit allocation loaded" << std::endl;
    }

    // encoding phase
    START_TIMING(VAQ_TRAIN);
    if (args["save"] == "" || !isFileExists(args["save"])) {
      std::cout << "Training the centroids" << std::endl;
      vaq.train(dataset, true);
    }
    END_TIMING(VAQ_TRAIN, "== Training time: ");

    START_TIMING(VAQ_ENCODE);
    if (args["save-enc"] != "" && isFileExists(args["save-enc"])) {
      if (args["save"] == "") {
        std::cout << "Attempt to read encoded dataset without reading saved centroids" << std::endl;
        std::cout << "Exiting" << std::endl;
        exit(0);
      }
      std::cout << "using saved encoded dataset" << std::endl;
      vaq.mCodebook = loadCodebook<CodebookType>(args["save-enc"]);
    } else {
      vaq.encode(dataset);
    }

    if ((vaq.searchMethod() & VAQ::NNMethod::Fast) || (vaq.searchMethod() & VAQ::NNMethod::Fast3)) {
      START_TIMING(LEARN_QUANTIZATION);
      vaq.learnQuantization(dataset, args.at<float>("learn-ratio"));
      END_TIMING(LEARN_QUANTIZATION, "== Learn Quantization time: ");
    }
    END_TIMING(VAQ_ENCODE, "== Encoding time: ");

    dataset.resize(0, 0); // release dataset memory

    // Maybe for next project 
    #if 0
    if (false) {
      // find frequent pattern
      START_TIMING(FIND_PATTERN);
      SetPattern patterns = findFrequentPattern(
        vaq.mCodebook.block(0, 0, 10000, vaq.mCodebook.cols()), 
        150,
        vaq.mCentroidsNum
      );
      END_TIMING(FIND_PATTERN, "== Find pattern time: ");
      std::cout << "pattern found: " << patterns.size() << std::endl;
      std::vector<Pattern> filteredPatterns(patterns.begin(), patterns.end());
      
      // Filter pattern size
      uint min_pattern_size = 2;
      // while (filteredPatterns.size() > vaq.mTIClusterNum) {
        filteredPatterns.erase(std::remove_if(
          filteredPatterns.begin(), filteredPatterns.end(),
          [min_pattern_size](const Pattern & x){ return x.first.size() < min_pattern_size; }
        ), filteredPatterns.end());
      //   min_pattern_size += 1;
      //   if (min_pattern_size >= vaq.mCodebook.cols()) {
      //     break;
      //   }
      // }

      // Remove subset
      {
        std::vector<Pattern> tempFiltered;
        tempFiltered.reserve(filteredPatterns.size());
        for (int i=0; i<(int)filteredPatterns.size(); i++) {
          bool filtered = false;
          for (const auto &p: filteredPatterns) {
            if (filteredPatterns[i].first.size() != p.first.size()) {
              std::set<Item> diff;
              std::set_difference(
                p.first.begin(), p.first.end(), 
                filteredPatterns[i].first.begin(), filteredPatterns[i].first.end(),
                std::inserter(diff, diff.begin())
              );
              if (diff.size() == 0) {
                filtered = true;
                break;
              }
            }
          }
          if (!filtered) {
            tempFiltered.push_back(filteredPatterns[i]);
          }
        }
        filteredPatterns = tempFiltered;
      }

      std::sort(filteredPatterns.begin(), filteredPatterns.end(),
        [](Pattern const& lhs, Pattern const& rhs) -> bool{
          // if (lhs.first.size() == rhs.first.size()) {
          //   return lhs.second > rhs.second;
          // }
          // return lhs.first.size() > rhs.first.size();
          if (lhs.second == rhs.second) {
            return lhs.first.size() > rhs.first.size();
          }
          return lhs.second > rhs.second;
        }
      );
      std::cout << "filtered pattern found: " << filteredPatterns.size() << std::endl;
      int ct = 0;
      for (const auto &p: filteredPatterns) {
        if (p.first.size() > 1) {
          std::cout << "[";
          for (const auto &i: p.first) {
              std::cout << i << ',';
          }
          std::cout << "] = " << ((float)p.second/vaq.mCodebook.rows() * 100) << " %" << std::endl;
          if (++ct >= 10) {
            break;
          }
        }
      }

      // use patterns to construct TI cluster
      // {
      //   vaq.mTIClusterNum = std::min((int)filteredPatterns.size(), vaq.mTIClusterNum);
      //   vaq.mTIClusters.resize(vaq.mTIClusterNum, vaq.mTotalDim);
      //   vaq.mTIClusters.setConstant(std::numeric_limits<float>::max());
        
      //   RowVector<int> offsets(vaq.mCentroidsNum.size());
      //   offsets(0) = 0;
      //   for (int i=1; i<(int)vaq.mCentroidsNum.size(); i++) {
      //       offsets(i) = vaq.mCentroidsNum[i] + offsets(i-1);
      //   }
      //   auto decodeitem = [&offsets](int item, int &belong) -> int {
      //     int i;
      //     for (i=1; i<offsets.size(); i++) {
      //       if (item < offsets(i)) {
      //         belong = i-1;
      //         return item - offsets(i-1);
      //       }
      //     }
      //     belong = i-1;
      //     return item - offsets(i-1);
      //   };
      //   int rowct = 0;
      //   for (const Pattern& p: filteredPatterns) {
      //     std::vector<int> items(p.first.begin(), p.first.end());
      //     for (auto it: items) {
      //       int belong;
      //       int decoded = decodeitem(it, belong);
      //       vaq.mTIClusters.row(rowct).segment(
      //         vaq.mSubsBeginIdx[belong], vaq.mSubsLen[belong]
      //       ) = vaq.mCentroidsPerSubs[belong].row(decoded);
      //     }
      //     for (int subs=0; subs<vaq.mCodebook.cols(); subs++) { 
      //       if (vaq.mTIClusters(rowct, vaq.mSubsBeginIdx[subs]) == std::numeric_limits<float>::max()) {
      //         int randIdx = rand() % vaq.mCentroidsNum[subs];
      //         vaq.mTIClusters.row(rowct).segment(
      //           vaq.mSubsBeginIdx[subs], vaq.mSubsLen[subs]
      //         ) = vaq.mCentroidsPerSubs[subs].row(randIdx);
      //       }
      //     }
      //     if (++rowct >= vaq.mTIClusterNum) {
      //       break;
      //     }
      //   }
      // }

      // std::cout << "clusters created: " << vaq.mTIClusters.row(0) << std::endl;

      // exit(0);

    }
    #endif

    if (vaq.searchMethod() & VAQ::NNMethod::TI) {
      START_TIMING(TI_CLUSTER);
      vaq.clusterTI(true, true);
      END_TIMING(TI_CLUSTER, "== TI Clustering time: ");
    }

    if (args["save"] != "" && !isFileExists(args["save"])) {
      std::cout << "Saving centroids to " << args["save"] << std::endl;
      saveCentroids(vaq.mCentroidsPerSubs, args["save"]);
    }

    if (args["save-enc"] != "" && !isFileExists(args["save-enc"])) {
      std::cout << "Saving codebook to " << args["save-enc"] << std::endl;
      saveCodebook(vaq.mCodebook, args["save-enc"]);
    }
  }

  {
    RowMatrixXf queries = RowMatrixXf::Zero(args.at<int>("queries-size"), args.at<int>("timeseries-size") + dimPadding);

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

    std::vector<std::vector<int>> topnn;
    if (args["groundtruth"] != "") {
      std::cout << "Read groundtruth" << std::endl;
      if (args["groundtruth-format"] == "ascii") {
        readTOPNNExternal(args["groundtruth"], topnn, args.at<int>("k"), ',');
      } else if (args["groundtruth-format"] == "ivecs") {
        readIVecsFromExternal(args["groundtruth"], topnn, args.at<int>("k"));
      } else if (args["groundtruth-format"] == "bin") {
        readTOPNNExternalBin(args["groundtruth"], topnn, args.at<int>("k"));
      }
    }

    std::cout << "Querying phase" << std::endl;
    std::vector<int> refines;
    if (args["refine"] != "") {
      std::stringstream ss(args["refine"]);
      while (ss.good())
      {
        std::string substr;
        getline(ss, substr, ',');
        refines.push_back(std::stoi(substr));
      }
    } else {
      refines.push_back(0);
    }

    RowMatrixXf datasetrefine;
    if (refines.size() > 1 || refines[0] > 0) {
      datasetrefine.resize(args.at<int>("dataset-size"), args.at<int>("timeseries-size") + dimPadding);
      datasetrefine.setZero();
      std::cout << "Read refining dataset" << std::endl;
      if (args["file-format-ori"] == "ascii") {
        readOriginalFromExternal<true>(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), ',');
      } else if (args["file-format-ori"] == "fvecs") {
        readFVecsFromExternal(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
      } else if (args["file-format-ori"] == "bvecs") {
        readBVecsFromExternal(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
      } else if (args["file-format-ori"] == "bin") {
        readFromExternalBin(args["dataset"], datasetrefine, args.at<int>("timeseries-size"), args.at<int>("dataset-size"));
      }
    }

    for (const int refine: refines) {
      START_TIMING(QUERY);                                                                                                                                                                    
      int searchK = refine >= args.at<int>("k") ? refine : args.at<int>("k");
      LabelDistVecF answers = vaq.search(queries, searchK, true);                                                                                                                


      if (refine >= args.at<int>("k")) {
        std::cout << "Refining the answer with Refine = " << refine << std::endl;
        answers = vaq.refine(queries, answers, datasetrefine, args.at<int>("k"));
      }
      END_TIMING(QUERY, "== Querying time: ");
      
      if (args["result"] != "") {
        std::string resultFP = args["result"];
        if (refines.size() > 1) {
          resultFP.append("_R" + std::to_string(refine));
        }
        std::cout << "Writing knn results to " << resultFP << std::endl;
        writeKNNResults(resultFP, answers, queries.rows());;
      }
      if (args["groundtruth"] != "") {
        // measure precision
        std::cout << "\tprecision(avg_recall): " << getAvgRecall(answers.labels, topnn, args.at<int>("k")) << std::endl;
        std::cout << "\trecall@R: " << getRecallAtR(answers.labels, topnn, args.at<int>("k")) << std::endl;
        std::cout << "\tMAP: " << getMeanAveragePrecision(answers.labels, topnn, args.at<int>("k")) << std::endl;
      }

    }
    

  }

  return 0;
}
