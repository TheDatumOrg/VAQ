#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <random>
#include <stdint.h>

struct RandomGenerator {
  std::mt19937 mt;

  // generate random integer
  int randomInt(int max) {
    return mt() % max;
  }

  explicit RandomGenerator (int seed = 13517106) : mt((unsigned int)seed) {}
};

void randomPermutation(std::vector<int> &out, int seed=13517106) {
  const size_t n = out.size();
  for (size_t i=0; i<n; i++) {
    out[i] = i;
  }
  RandomGenerator rng(seed);
  for (size_t i=0; i < n-1; i++) {
    int i2 = i + rng.randomInt(n-i);
    std::swap(out[i], out[i2]);
  }
}

/**
 * @brief Generate vector of random integer 
 * 
 * @param size vector size
 * @param start lower limit (inclusive)
 * @param end upper limit (inclusive)
 * @return std::vector<int> vector of random integer 
 */
std::vector<int> generateRandomIntVec(int size, int start, int end) {
  std::random_device device;
  std::mt19937 mersenne_engine {device()};
  std::uniform_int_distribution<int> dist {start, end};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::vector<int> vec(size);
  std::generate(std::begin(vec), std::end(vec), gen);

  return vec;
}

#endif
