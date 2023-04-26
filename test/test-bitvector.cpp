#include "catch.hpp"

#include <vector>
#include <array>
#include "BitVector.hpp"

TEST_CASE( "Bit vector types are tested", "[bitvector]" ) {
  REQUIRE(std::is_same<bitvectors, std::vector<bitv>>::value);
}
