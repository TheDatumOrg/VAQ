#include "catch.hpp"

#include <vector>
#include "BitVecEngine.hpp"

bool compareEqual(bitv v1, bitv v2) {
  assert(v1.size() == v2.size());
  for (int i=0; i<v1.size(); i++) {
    if (v1[i] != v2[i]) return false;
  }

  return true;
}

TEST_CASE( "BitVecEngine is tested", "[bitvecengine]" ) {
  {
    // 1 bit
    bitvectors bv;

    BitVecEngine engine(1);
    REQUIRE(engine.N == 1);
    REQUIRE(engine.data.size() == 0);
    
    engine.loadBitV(bv);
    REQUIRE(engine.data.size() == 0);

    BitVecEngine::generateDummyBitVectors(1, bv, 3, 1);
    REQUIRE(bv.size() == 3);
    engine.loadBitV(bv);
    REQUIRE(engine.data.size() == 3);
    REQUIRE(engine.getBitV(0) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(1) == createBitV(1, {0}));
    REQUIRE(engine.getBitV(2) == createBitV(1, {1}));

    engine.deleteBitV(1);
    REQUIRE(engine.data.size() == 2);
    REQUIRE(engine.getBitV(0) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(1) == createBitV(1, {1}));

    bitvectors bv2;
    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 2);

    bv2.push_back(createBitV(1, {1}));
    REQUIRE(engine.data.size() == 2);

    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 3);
    REQUIRE(engine.getBitV(0) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(1) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(2) == createBitV(1, {1}));

    bv2.clear();
    bv2.push_back(createBitV(1, {0}));
    bv2.push_back(createBitV(1, {0}));
    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 5);
    REQUIRE(engine.getBitV(0) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(1) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(2) == createBitV(1, {1}));
    REQUIRE(engine.getBitV(3) == createBitV(1, {0}));
    REQUIRE(engine.getBitV(4) == createBitV(1, {0}));

    bitvectors qHamming1;
    qHamming1.push_back(engine.getBitV(0));
    
    std::vector<std::vector<IdxDistPair>> pHamming1 = engine.query(qHamming1, 1);
    REQUIRE(pHamming1.size() == 1);
    REQUIRE(pHamming1.at(0).size() == 1);
    REQUIRE(pHamming1.at(0)[0].idx == 0);
    
    bitvectors qHamming2;
    qHamming2.push_back(engine.getBitV(1));
    std::vector<std::vector<IdxDistPair>> pHamming2 = engine.query(qHamming2, 3);
    REQUIRE(pHamming2.size() == 1);
    REQUIRE(pHamming2.at(0).size() == 3);
    REQUIRE(pHamming2.at(0)[0].idx == 0);
    REQUIRE(pHamming2.at(0)[1].idx == 1);
    REQUIRE(pHamming2.at(0)[2].idx == 2);

    // bitvectors qJaccard1;
    // qJaccard1.push_back(engine.getBitV(0));
    // std::vector<std::vector<IdxDistPair>> pJaccard1 = engine.query(qJaccard1, 1, jaccardDist<1>);
    // REQUIRE(pJaccard1.size() == 1);
    // REQUIRE(pJaccard1.at(0).size() == 1);
    // REQUIRE(pJaccard1.at(0)[0].idx == 0);
    
    // bitvectors qJaccard2;
    // qJaccard2.push_back(engine.getBitV(1));
    // std::vector<std::vector<IdxDistPair>> pJaccard2 = engine.query(qJaccard2, 3, jaccardDist<1>);
    // REQUIRE(pJaccard2.size() == 1);
    // REQUIRE(pJaccard2.at(0).size() == 3);
    // REQUIRE(pJaccard2.at(0)[0].idx == 0);
    // REQUIRE(pJaccard2.at(0)[1].idx == 1);
    // REQUIRE(pJaccard2.at(0)[2].idx == 2);


    // Testing parallel query
    bitvectors qParallel;
    qParallel.push_back(engine.getBitV(0));
    qParallel.push_back(engine.getBitV(3));

    std::vector<std::vector<IdxDistPair>> pParallel1 = engine.queryParallel(qParallel, 2, 1);
    std::vector<std::vector<IdxDistPair>> pParallel2 = engine.queryParallel(qParallel, 2, 2);

    REQUIRE(pParallel1.at(0)[0].idx == pParallel2.at(0)[0].idx);
    REQUIRE(pParallel1.at(0)[1].idx == pParallel2.at(0)[1].idx);
    REQUIRE(pParallel1.at(0)[0].dist == pParallel2.at(0)[0].dist);
    REQUIRE(pParallel1.at(0)[1].dist == pParallel2.at(0)[1].dist);
    REQUIRE(pParallel1.at(0)[0].idx == pParallel2.at(0)[0].idx);
    REQUIRE(pParallel1.at(1)[1].idx == pParallel2.at(1)[1].idx);
    REQUIRE(pParallel1.at(0)[0].dist == pParallel2.at(0)[0].dist);
    REQUIRE(pParallel1.at(1)[1].dist == pParallel2.at(1)[1].dist);
  }

  {
    // 32 bit
    bitvectors bv;

    BitVecEngine engine(32);
    REQUIRE(engine.N == 32);
    REQUIRE(engine.data.size() == 0);
    
    engine.loadBitV(bv);
    REQUIRE(engine.data.size() == 0);

    BitVecEngine::generateDummyBitVectors(32, bv, 3, 1);
    REQUIRE(bv.size() == 3);
    engine.loadBitV(bv);
    REQUIRE(engine.data.size() == 3);

    REQUIRE(engine.getBitV(0) == createBitV(32, {0x6B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(32, {0x327B23C6}));
    REQUIRE(engine.getBitV(2) == createBitV(32, {0x643C9869}));

    engine.deleteBitV(1);
    REQUIRE(engine.data.size() == 2);
    REQUIRE(engine.getBitV(0) == createBitV(32, {0x6B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(32, {0x643C9869}));

    bitvectors bv2;
    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 2);

    bv2.push_back(createBitV(32, {0xFFFFFFF0}));
    REQUIRE(engine.data.size() == 2);

    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 3);
    REQUIRE(engine.getBitV(0) == createBitV(32, {0x6B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(32, {0x643C9869}));
    REQUIRE(engine.getBitV(2) == createBitV(32, {0xFFFFFFF0}));

    bv2.clear();
    bv2.push_back(createBitV(32, {0xF0000000}));
    bv2.push_back(createBitV(32, {0x0000000F}));
    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 5);
    REQUIRE(engine.getBitV(0) == createBitV(32, {0x6B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(32, {0x643C9869}));
    REQUIRE(engine.getBitV(2) == createBitV(32, {0xFFFFFFF0}));
    REQUIRE(engine.getBitV(3) == createBitV(32, {0xF0000000}));
    REQUIRE(engine.getBitV(4) == createBitV(32, {0x0000000F}));

    bitvectors qHamming1;
    qHamming1.push_back(engine.getBitV(0));
    std::vector<std::vector<IdxDistPair>> pHamming1 = engine.query(qHamming1, 1);
    REQUIRE(pHamming1.size() == 1);
    REQUIRE(pHamming1.at(0).size() == 1);
    REQUIRE(pHamming1.at(0)[0].idx == 0);
    
    bitvectors qHamming2;
    qHamming2.push_back(engine.getBitV(1));
    std::vector<std::vector<IdxDistPair>> pHamming2 = engine.query(qHamming2, 3);
    REQUIRE(pHamming2.size() == 1);
    REQUIRE(pHamming2.at(0).size() == 3);
    REQUIRE(pHamming2.at(0)[0].idx == 1);
    REQUIRE(pHamming2.at(0)[1].idx == 3);
    REQUIRE(pHamming2.at(0)[2].idx == 4);

    // bitvectors qJaccard1;
    // qJaccard1.push_back(engine.getBitV(0));
    // std::vector<std::vector<IdxDistPair>> pJaccard1 = engine.query(qJaccard1, 1, jaccardDist<32>);
    // REQUIRE(pJaccard1.size() == 1);
    // REQUIRE(pJaccard1.at(0).size() == 1);
    // REQUIRE(pJaccard1.at(0)[0].idx == 0);
    
    // bitvectors qJaccard2;
    // qJaccard2.push_back(engine.getBitV(1));
    // std::vector<std::vector<IdxDistPair>> pJaccard2 = engine.query(qJaccard2, 3, jaccardDist<32>);
    // REQUIRE(pJaccard2.size() == 1);
    // REQUIRE(pJaccard2.at(0).size() == 3);
    // REQUIRE(pJaccard2.at(0)[0].idx == 1);
    // REQUIRE(pJaccard2.at(0)[1].idx == 2);
    // REQUIRE(pJaccard2.at(0)[2].idx == 0);
  }

  {
    // 64 bit
    bitvectors bv;

    BitVecEngine engine(64);
    REQUIRE(engine.N == 64);
    REQUIRE(engine.data.size() == 0);
    
    engine.loadBitV(bv);
    REQUIRE(engine.data.size() == 0);

    BitVecEngine::generateDummyBitVectors(64, bv, 3, 1);
    REQUIRE(bv.size() == 3);
    engine.loadBitV(bv);
    REQUIRE(engine.data.size() == 3);
    REQUIRE(engine.getBitV(0) == createBitV(64, {0x327B23C66B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(64, {0x66334873643C9869}));
    REQUIRE(engine.getBitV(2) == createBitV(64, {0x19495CFF74B0DC51}));

    engine.deleteBitV(1);
    REQUIRE(engine.data.size() == 2);
    REQUIRE(engine.getBitV(0) == createBitV(64, {0x327B23C66B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(64, {0x19495CFF74B0DC51}));

    bitvectors bv2;
    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 2);

    bv2.push_back(createBitV(64, {0xFFFFFFF0FFFFFFFF}));
    REQUIRE(engine.data.size() == 2);

    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 3);
    REQUIRE(engine.getBitV(0) == createBitV(64, {0x327B23C66B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(64, {0x19495CFF74B0DC51}));
    REQUIRE(engine.getBitV(2) == createBitV(64, {0xFFFFFFF0FFFFFFFF}));

    bv2.clear();
    bv2.push_back(createBitV(64, {0x00000000F0000000}));
    bv2.push_back(createBitV(64, {0x000000000000000F}));
    engine.appendBitV(bv2);
    REQUIRE(engine.data.size() == 5);
    REQUIRE(engine.getBitV(0) == createBitV(64, {0x327B23C66B8B4567}));
    REQUIRE(engine.getBitV(1) == createBitV(64, {0x19495CFF74B0DC51}));
    REQUIRE(engine.getBitV(2) == createBitV(64, {0xFFFFFFF0FFFFFFFF}));
    REQUIRE(engine.getBitV(3) == createBitV(64, {0x00000000F0000000}));
    REQUIRE(engine.getBitV(4) == createBitV(64, {0x000000000000000F}));

    bitvectors qHamming1;
    qHamming1.push_back(engine.getBitV(0));
    std::vector<std::vector<IdxDistPair>> pHamming1 = engine.query(qHamming1, 1);
    REQUIRE(pHamming1.size() == 1);
    REQUIRE(pHamming1.at(0).size() == 1);
    REQUIRE(pHamming1.at(0)[0].idx == 0);
    
    bitvectors qHamming2;
    qHamming2.push_back(engine.getBitV(1));
    std::vector<std::vector<IdxDistPair>> pHamming2 = engine.query(qHamming2, 3);
    REQUIRE(pHamming2.size() == 1);
    REQUIRE(pHamming2.at(0).size() == 3);
    REQUIRE(pHamming2.at(0)[0].idx == 1);
    REQUIRE(pHamming2.at(0)[1].idx == 3);
    REQUIRE(pHamming2.at(0)[2].idx == 2);

    // bitvectors qJaccard1;
    // qJaccard1.push_back(engine.getBitV(0));
    // std::vector<std::vector<IdxDistPair>> pJaccard1 = engine.query(qJaccard1, 1, jaccardDist<64>);
    // REQUIRE(pJaccard1.size() == 1);
    // REQUIRE(pJaccard1.at(0).size() == 1);
    // REQUIRE(pJaccard1.at(0)[0].idx == 0);
    
    // bitvectors qJaccard2;
    // qJaccard2.push_back(engine.getBitV(1));
    // std::vector<std::vector<IdxDistPair>> pJaccard2 = engine.query(qJaccard2, 3, jaccardDist<64>);
    // REQUIRE(pJaccard2.size() == 1);
    // REQUIRE(pJaccard2.at(0).size() == 3);
    // REQUIRE(pJaccard2.at(0)[0].idx == 1);
    // REQUIRE(pJaccard2.at(0)[1].idx == 2);
    // REQUIRE(pJaccard2.at(0)[2].idx == 0);
  }

}

TEST_CASE ("Progressive filtering with sort is tested", "[bitvecengine]") {
  {
    // N = 256, SubVector = 1
    const int N = 256;
    bitvectors bv;
    BitVecEngine engine(N);
    
    bv.push_back(createBitV(N, {0, 1, 3, 7}));
    bv.push_back(createBitV(N, {1, 1, 3, 7}));
    engine.loadBitV(bv);

    bitvectors queries;
    queries.push_back(createBitV(N, {1, 1, 3, 7}));
    std::vector<std::vector<IdxDistPair>> p = engine.queryFiltering_Sort(queries, 1);
    
    REQUIRE(p.size() == 1);
    REQUIRE(p.at(0).size() == 1);
    REQUIRE(p.at(0)[0].idx == 1);

  }

  {
    // N = 256, SubVector = 1
    const int N = 256;
    bitvectors bv;
    BitVecEngine engine(N);
    
    bv.push_back(createBitV(N, {0, 1, 3, 7}));
    bv.push_back(createBitV(N, {1, 1, 3, 7}));
    engine.loadBitV(bv);

    bitvectors queries;
    queries.push_back(createBitV(N, {1, 1, 3, 7}));
    std::vector<std::vector<IdxDistPair>> p = engine.queryFiltering_Sort(queries, 1);
    
    REQUIRE(p.size() == 1);
    REQUIRE(p.at(0).size() == 1);
    REQUIRE(p.at(0)[0].idx == 1);
  }

}

TEST_CASE ("Progressive filtering is tested", "[bitvecengine]") {
  // test minimum of N=256
  // current constraint: N multiply of 4 (due to avx2)
  {
    // N = 256, SubVector = 1
    const int N = 256;
    const int M = 1;
    bitvectors bv;
    BitVecEngine engine(N, M);
    
    bv.push_back(createBitV(N, {0, 1, 3, 7}));
    bv.push_back(createBitV(N, {1, 1, 3, 7}));
    engine.loadBitV(bv);

    bitvectors queries;
    queries.push_back(createBitV(N, {1, 1, 3, 7}));
    std::vector<std::vector<IdxSubDistPair>> p = engine.queryFiltering_Heap(queries, 1);
    
    REQUIRE(p.size() == 1);
    REQUIRE(p.at(0).size() == 1);
    REQUIRE(p.at(0)[0].idx == 1);
  }

  {
    // N = 512, SubVector = 2
    const int N = 512;
    const int M = 2;
    bitvectors bv;
    BitVecEngine engine(N, M);
    
    bv.push_back(createBitV(N, {1, 1, 3, 7, 0, 1, 3, 7}));
    bv.push_back(createBitV(N, {1, 1, 3, 7, 1, 1, 3, 7}));
    bv.push_back(createBitV(N, {1, 1, 3, 7, 2, 1, 3, 7}));
    bv.push_back(createBitV(N, {1, 1, 3, 7, 1, 1, 3, 7}));
    engine.loadBitV(bv);

    bitvectors queries;
    queries.push_back(createBitV(N, {1, 1, 3, 7, 1, 1, 3, 7}));
    std::vector<std::vector<IdxSubDistPair>> p = engine.queryFiltering_Heap(queries, 4);
    
    REQUIRE(p.size() == 1);
    REQUIRE(p.at(0).size() == 4);
    REQUIRE(p.at(0)[0].idx == 3);
    REQUIRE(p.at(0)[1].idx == 1);
    REQUIRE(p.at(0)[2].idx == 0);
    REQUIRE(p.at(0)[3].idx == 2);
  }
}
