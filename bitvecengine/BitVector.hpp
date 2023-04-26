#ifndef BITVECTOR_H_
#define BITVECTOR_H_

#include <vector>
#include <assert.h>
#include <initializer_list>
#include <stdint.h>
#include <algorithm>

/**
 * @brief BitV generic type
 */
using bitv = typename std::vector<uint64_t>;

/**
 * @brief bitvectors wrapped by vector STL
 * 
 */
using bitvectors = typename std::vector<bitv>;

inline uint64_t LSB(int N) {
  uint64_t mask = 0;
  for (int i=0; i<N-1; i++) {
    mask |= 1;
    mask <<= 1;
  }
  mask |= 1;

  return mask;
}

inline uint64_t MSB(int N) {
  return !LSB(64-N);
}

inline int actualBitVLen(int N) {
  return (N + 63) / 64;
}

inline void printBitV(const bitv v) {
  printf("actual Length = %lu\n", v.size());
  for (int i=0; i<(int)v.size(); i++) {
    printf("0x%X%X ",(uint32_t)(v[i] >> 32), (uint32_t)v[i]);
  }
  printf("\n");
}

inline bitv createBitV(const int N, uint64_t raw) {
  bitv v(actualBitVLen(N), 0);
  
  const int actualLength = v.size();
  if (N <= 64) {
    v[0] = raw;
  } else {
    for (int i=0; i<actualLength-1; i++) {
      v[i] = (raw >> 64*(actualLength-(i+1))) & LSB(64);
    }
    v[actualLength-1] = raw & LSB(N - (actualLength-1) * 64);
  }

  return v;
}

template <class T>
inline bitv createBitV(const int N, std::initializer_list<T> list) {
  assert(list.size() == (uint)actualBitVLen(N));
  bitv v(actualBitVLen(N), 0);
  
  int i=0;
  for (auto elem: list) {
    v[i] = elem;
    i++;
  }

  return v;
}


#endif
