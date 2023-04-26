#include "catch.hpp"

#include <limits>
#include <cfloat>
#include "BitVecEngine.hpp"

bool approxEqual(float a, float b) {
  return fabs(a-b) < FLT_EPSILON;
}

TEST_CASE( "Hamming distance is tested", "[distancefunction]" ) {

  // Test interchangable params
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x0), createBitV(4, 0x1)), 1.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x1), createBitV(4, 0x0)), 1.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x0), createBitV(4, 0xF)), 4.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0xF), createBitV(4, 0x0)), 4.0f));
  
  // 4 bit
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x0), createBitV(4, 0x0)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x8), createBitV(4, 0x8)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0xF), createBitV(4, 0xF)), 0.0f));

  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x0), createBitV(4, 0x3)), 2.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x0), createBitV(4, 0x7)), 3.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(4, 0x0), createBitV(4, 0xF)), 4.0f));
  
  // 8 bit
  REQUIRE(approxEqual(hammingDist(createBitV(8, 0x00), createBitV(8, 0x00)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(8, 0x0F), createBitV(8, 0x0F)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(8, 0xFF), createBitV(8, 0xFF)), 0.0f));

  REQUIRE(approxEqual(hammingDist(createBitV(8, 0x00), createBitV(8, 0x03)), 2.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(8, 0x00), createBitV(8, 0x1E)), 4.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(8, 0x00), createBitV(8, 0xFF)), 8.0f));

  // 16 bit
  REQUIRE(approxEqual(hammingDist(createBitV(16, 0x0000), createBitV(16, 0x0000)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(16, 0x00FF), createBitV(16, 0x00FF)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(16, 0xFFFF), createBitV(16, 0xFFFF)), 0.0f));
  
  REQUIRE(approxEqual(hammingDist(createBitV(16, 0x0000), createBitV(16, 0x0003)), 2.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(16, 0x0000), createBitV(16, 0x00FF)), 8.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(16, 0x0000), createBitV(16, 0xFFFF)), 16.0f));

  // 32 bit
  REQUIRE(approxEqual(hammingDist(createBitV(32, 0x00000000), createBitV(32, 0x00000000)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(32, 0x0000FFFF), createBitV(32, 0x0000FFFF)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(32, 0xFFFFFFFF), createBitV(32, 0xFFFFFFFF)), 0.0f));
  
  REQUIRE(approxEqual(hammingDist(createBitV(32, 0x00000000), createBitV(32, 0x00000003)), 2.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(32, 0x00000000), createBitV(32, 0x0000FFFF)), 16.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(32, 0x00000000), createBitV(32, 0xFFFFFFFF)), 32.0f));

  // 64 bit
  REQUIRE(approxEqual(hammingDist(createBitV(64, 0x0000000000000000), createBitV(64, 0x0000000000000000)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(64, 0x00000000FFFFFFFF), createBitV(64, 0x00000000FFFFFFFF)), 0.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(64, 0xFFFFFFFFFFFFFFFF), createBitV(64, 0xFFFFFFFFFFFFFFFF)), 0.0f));
  
  REQUIRE(approxEqual(hammingDist(createBitV(64, 0x0000000000000000), createBitV(64, 0x0000000000000003)), 2.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(64, 0x0000000000000000), createBitV(64, 0x00000000FFFFFFFF)), 32.0f));
  REQUIRE(approxEqual(hammingDist(createBitV(64, 0x0000000000000000), createBitV(64, 0xFFFFFFFFFFFFFFFF)), 64.0f));
}

TEST_CASE( "Jaccard distance is tested", "[distancefunction]" ) {
  // Test interchangable params
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x0), createBitV(4, 0x1)), 1.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x1), createBitV(4, 0x0)), 1.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x0), createBitV(4, 0xF)), 1.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0xF), createBitV(4, 0x0)), 1.0f));
  
  // 4 bit
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x0), createBitV(4, 0x0)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x8), createBitV(4, 0x8)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0xF), createBitV(4, 0xF)), 0.0f));

  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x1), createBitV(4, 0x3)), 0.5f));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x1), createBitV(4, 0x7)), (1.0f-1.0f/3)));
  REQUIRE(approxEqual(jaccardDist(createBitV(4, 0x1), createBitV(4, 0xF)), 0.75f));
  
  // 8 bit
  REQUIRE(approxEqual(jaccardDist(createBitV(8, 0x00), createBitV(8, 0x00)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(8, 0x0F), createBitV(8, 0x0F)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(8, 0xFF), createBitV(8, 0xFF)), 0.0f));

  REQUIRE(approxEqual(jaccardDist(createBitV(8, 0x11), createBitV(8, 0x03)), 2.0f/3));
  REQUIRE(approxEqual(jaccardDist(createBitV(8, 0x11), createBitV(8, 0x1E)), 4.0f/5));
  REQUIRE(approxEqual(jaccardDist(createBitV(8, 0x11), createBitV(8, 0xFF)), 6.0f/8));

  // 16 bit
  REQUIRE(approxEqual(jaccardDist(createBitV(16, 0x0000), createBitV(16, 0x0000)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(16, 0x00FF), createBitV(16, 0x00FF)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(16, 0xFFFF), createBitV(16, 0xFFFF)), 0.0f));
  
  REQUIRE(approxEqual(jaccardDist(createBitV(16, 0x1111), createBitV(16, 0x0003)), 4.0f/5));
  REQUIRE(approxEqual(jaccardDist(createBitV(16, 0x1111), createBitV(16, 0x00FF)), 4.0f/5));
  REQUIRE(approxEqual(jaccardDist(createBitV(16, 0x1111), createBitV(16, 0xFFFF)), 3.0f/4));

  // 32 bit
  REQUIRE(approxEqual(jaccardDist(createBitV(32, 0x00000000), createBitV(32, 0x00000000)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(32, 0x0000FFFF), createBitV(32, 0x0000FFFF)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(32, 0xFFFFFFFF), createBitV(32, 0xFFFFFFFF)), 0.0f));
  
  REQUIRE(approxEqual(jaccardDist(createBitV(32, 0x11111111), createBitV(32, 0x00000003)), 8.0f/9));
  REQUIRE(approxEqual(jaccardDist(createBitV(32, 0x11111111), createBitV(32, 0x0000FFFF)), 4.0f/5));
  REQUIRE(approxEqual(jaccardDist(createBitV(32, 0x11111111), createBitV(32, 0xFFFFFFFF)), 3.0f/4));

  // 64 bit
  REQUIRE(approxEqual(jaccardDist(createBitV(64, 0x00000000), createBitV(64, 0x00000000)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(64, 0x0000FFFF), createBitV(64, 0x0000FFFF)), 0.0f));
  REQUIRE(approxEqual(jaccardDist(createBitV(64, 0xFFFFFFFF), createBitV(64, 0xFFFFFFFF)), 0.0f));
  
  REQUIRE(approxEqual(jaccardDist(createBitV(64, 0x11111111), createBitV(64, 0x00000003)), 8.0f/9));
  REQUIRE(approxEqual(jaccardDist(createBitV(64, 0x11111111), createBitV(64, 0x0000FFFF)), 4.0f/5));
  REQUIRE(approxEqual(jaccardDist(createBitV(64, 0x11111111), createBitV(64, 0xFFFFFFFF)), 3.0f/4));
}

TEST_CASE( "Hamming distance with progressive filtering", "[distancefunction]") {
  // simple test
  const int N = 256;
  const int M = 1;
  bitv q = createBitV(N, {1u, 1u, 3u, 7u});
  
  bitv a = createBitV(N, {0u, 1u, 3u, 7u});
  bitv b = createBitV(N, {1u, 1u, 3u, 7u});

  int r1 = hammingDistSub(q, a, M, 0);
  int r2 = hammingDistSub(q, b, M, 0);

  REQUIRE(r1 == 1);
  REQUIRE(r2 == 0);
}
