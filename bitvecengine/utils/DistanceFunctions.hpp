#ifndef __DISTANCE_FUNCTION_H_
#define __DISTANCE_FUNCTION_H_

#include <cmath>
#include <immintrin.h>
#include <cassert>
#include "../BitVector.hpp"

/**
 * Harley-Seal AVX population count function utility
 */

/**
 * Compute the population count of a 256-bit word
 * This is not especially fast, but it is convenient as part of other functions.
 */
static inline __m256i popcount256(__m256i v) {
    const __m256i lookuppos = _mm256_setr_epi8(
        /* 0 */ 4 + 0, /* 1 */ 4 + 1, /* 2 */ 4 + 1, /* 3 */ 4 + 2,
        /* 4 */ 4 + 1, /* 5 */ 4 + 2, /* 6 */ 4 + 2, /* 7 */ 4 + 3,
        /* 8 */ 4 + 1, /* 9 */ 4 + 2, /* a */ 4 + 2, /* b */ 4 + 3,
        /* c */ 4 + 2, /* d */ 4 + 3, /* e */ 4 + 3, /* f */ 4 + 4,

        /* 0 */ 4 + 0, /* 1 */ 4 + 1, /* 2 */ 4 + 1, /* 3 */ 4 + 2,
        /* 4 */ 4 + 1, /* 5 */ 4 + 2, /* 6 */ 4 + 2, /* 7 */ 4 + 3,
        /* 8 */ 4 + 1, /* 9 */ 4 + 2, /* a */ 4 + 2, /* b */ 4 + 3,
        /* c */ 4 + 2, /* d */ 4 + 3, /* e */ 4 + 3, /* f */ 4 + 4);
    const __m256i lookupneg = _mm256_setr_epi8(
        /* 0 */ 4 - 0, /* 1 */ 4 - 1, /* 2 */ 4 - 1, /* 3 */ 4 - 2,
        /* 4 */ 4 - 1, /* 5 */ 4 - 2, /* 6 */ 4 - 2, /* 7 */ 4 - 3,
        /* 8 */ 4 - 1, /* 9 */ 4 - 2, /* a */ 4 - 2, /* b */ 4 - 3,
        /* c */ 4 - 2, /* d */ 4 - 3, /* e */ 4 - 3, /* f */ 4 - 4,

        /* 0 */ 4 - 0, /* 1 */ 4 - 1, /* 2 */ 4 - 1, /* 3 */ 4 - 2,
        /* 4 */ 4 - 1, /* 5 */ 4 - 2, /* 6 */ 4 - 2, /* 7 */ 4 - 3,
        /* 8 */ 4 - 1, /* 9 */ 4 - 2, /* a */ 4 - 2, /* b */ 4 - 3,
        /* c */ 4 - 2, /* d */ 4 - 3, /* e */ 4 - 3, /* f */ 4 - 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo = _mm256_and_si256(v, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookuppos, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookupneg, hi);
    return _mm256_sad_epu8(popcnt1, popcnt2);
}

/**
 * Simple CSA over 256 bits
 */
static inline void CSA(__m256i *h, __m256i *l, __m256i a, __m256i b,
                       __m256i c) {
    const __m256i u = _mm256_xor_si256(a, b);
    *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    *l = _mm256_xor_si256(u, c);
}

/**
 * Fast Harley-Seal AVX population count function
 */
inline static uint64_t avx2_harley_seal_popcount256(const __m256i *data,
                                                    const uint64_t size) {
    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256();
    __m256i twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const uint64_t limit = size - size % 16;
    uint64_t i = 0;

    for (; i < limit; i += 16) {
        CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i),
            _mm256_lddqu_si256(data + i + 1));
        CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 2),
            _mm256_lddqu_si256(data + i + 3));
        CSA(&foursA, &twos, twos, twosA, twosB);
        CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i + 4),
            _mm256_lddqu_si256(data + i + 5));
        CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 6),
            _mm256_lddqu_si256(data + i + 7));
        CSA(&foursB, &twos, twos, twosA, twosB);
        CSA(&eightsA, &fours, fours, foursA, foursB);
        CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i + 8),
            _mm256_lddqu_si256(data + i + 9));
        CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 10),
            _mm256_lddqu_si256(data + i + 11));
        CSA(&foursA, &twos, twos, twosA, twosB);
        CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i + 12),
            _mm256_lddqu_si256(data + i + 13));
        CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 14),
            _mm256_lddqu_si256(data + i + 15));
        CSA(&foursB, &twos, twos, twosA, twosB);
        CSA(&eightsB, &fours, fours, foursA, foursB);
        CSA(&sixteens, &eights, eights, eightsA, eightsB);

        total = _mm256_add_epi64(total, popcount256(sixteens));
    }

    total = _mm256_slli_epi64(total, 4);  // * 16
    total = _mm256_add_epi64(
        total, _mm256_slli_epi64(popcount256(eights), 3));  // += 8 * ...
    total = _mm256_add_epi64(
        total, _mm256_slli_epi64(popcount256(fours), 2));  // += 4 * ...
    total = _mm256_add_epi64(
        total, _mm256_slli_epi64(popcount256(twos), 1));  // += 2 * ...
    total = _mm256_add_epi64(total, popcount256(ones));
    for (; i < size; i++)
        total =
            _mm256_add_epi64(total, popcount256(_mm256_lddqu_si256(data + i)));

    return (uint64_t)(_mm256_extract_epi64(total, 0)) +
           (uint64_t)(_mm256_extract_epi64(total, 1)) +
           (uint64_t)(_mm256_extract_epi64(total, 2)) +
           (uint64_t)(_mm256_extract_epi64(total, 3));
}

#define EXTRACTACCUMULATE(a) (uint64_t)(_mm256_extract_epi64(a, 0)) + \
                             (uint64_t)(_mm256_extract_epi64(a, 1)) + \
                             (uint64_t)(_mm256_extract_epi64(a, 2)) + \
                             (uint64_t)(_mm256_extract_epi64(a, 3))

/**
 * END Harley-Seal AVX population count function utility
 */

#define UNWRAP(a, offset) a[0+offset], a[1+offset], a[2+offset], a[3+offset]

/**
 * @brief Compute hamming distance using Harley-Seal
 * 
 * @tparam N bitv size
 * @param v1 bitv
 * @param v2 bitv
 * @return uint32_t distance
 */
inline uint32_t hammingDistHarleySeal(const bitv &v1, const bitv &v2) {
  uint32_t setBits = 0;

  const int actBitVLen = v1.size();
  __m256i x1, x2, popcnt;
  if (actBitVLen >= 4) {
    const int max = (actBitVLen / 4) * 4;
    for (int i=0; i<max; i += 4)  {
      x1 = _mm256_set_epi64x(UNWRAP(v1, i));
      x2 = _mm256_set_epi64x(UNWRAP(v2, i));
      popcnt = popcount256(_mm256_xor_si256(x1, x2));
      setBits += (uint32_t)(EXTRACTACCUMULATE(popcnt));
    }
  } 

  if (actBitVLen % 4 != 0) {
    const int startOffset = (actBitVLen / 4) *4;
    const int rest = actBitVLen % 4;
    for (int i=0; i<rest; i++) {
      setBits += __builtin_popcountl(v1[i+startOffset] ^ v2[i+startOffset]);
    }
  }
  
  return setBits;
}

inline uint32_t hammingDist(const bitv &v1, const bitv &v2) {
  uint32_t setBits = 0;

  const int actBitVLen = v1.size();
  for (int iter=0; iter<actBitVLen; iter++) {
    setBits += __builtin_popcountl(v1[iter] ^ v2[iter]);
  }
  return setBits;
}

inline uint32_t hammingDistEarlyAbandon(const bitv &v1, const bitv &v2, const uint32_t bsf) {
  uint32_t setBits = 0;

  const int actBitVLen = v1.size();
  for (int iter=0; iter<actBitVLen && setBits < bsf; iter++) {
    setBits += __builtin_popcountl(v1[iter] ^ v2[iter]);
  }
  return setBits;
}

inline uint32_t hammingDistSub(const bitv &v1, const bitv &v2, int subvectorLen, int subvectorIdx) {
  uint32_t setBits = 0;

  const int actBitVLen = v1.size();
  // assert((int)((float)actBitVLen / subvectorLen) == actBitVLen / subvectorLen);
  // assert((actBitVLen / subvectorLen)%4 == 0);  // TODO: remove this constrain

  // __m256i x1, x2, popcnt;
  // if (actBitVLen >= 4) {
    const int proportionalLen = actBitVLen/subvectorLen;
    const int min = subvectorIdx * proportionalLen;
    const int max = min + proportionalLen;
    for (int i=min; i<max; i++) {
      setBits += __builtin_popcount(v1[i] ^ v2[i]);
    }
  //   for (int i=min; i<max; i += 4)  {
  //     x1 = _mm256_set_epi64x(UNWRAP(v1, i));
  //     x2 = _mm256_set_epi64x(UNWRAP(v2, i));
  //     popcnt = popcount256(_mm256_xor_si256(x1, x2));
  //     setBits += (int32_t)(EXTRACTACCUMULATE(popcnt));
  //   }
  // }

  return setBits;
}

/**
 * @brief Compute Jaccard distance
 * 
 * @tparam N bitv size
 * @param v1 bitv
 * @param v2 bitv
 * @return float distance
 */
inline float jaccardDist(const bitv &v1, const bitv &v2) {
  int intersectSize = 0, unionSize = 0;
  const int actualLength = v1.size();
  for (int i=0; i<actualLength; i++)  {
    intersectSize += __builtin_popcountl(v1[i] & v2[i]);
    unionSize += __builtin_popcountl(v1[i] | v2[i]);
  }

  if (unionSize == 0)
    return 0.0f;
  else
    return 1.0f - ((float)intersectSize/unionSize);
}

inline float euclideanDistNoSQRT(const std::vector<float> &v1, const std::vector<float> &v2) {
  float distance = 0.0;
  for (int i=0; i<(int)v1.size(); i++) {
    distance += (v1[i]-v2[i]) * (v1[i]-v2[i]);
  }
  
  return distance;
}

inline float euclideanDist(const std::vector<float> &v1, const std::vector<float> &v2) {
  return sqrt(euclideanDistNoSQRT(v1, v2));
}

inline float euclideanDistEarlyAbandon(const std::vector<float> &v1, const std::vector<float> &v2, const float bsf) {
  float distance = 0.0;
  for (int i=0; i<(int)v1.size() && distance < bsf; i++) {
    distance += (v1[i]-v2[i]) * (v1[i]-v2[i]);
  }

  return distance;
}

inline float manhattanDist(const std::vector<float> &v1, const std::vector<float> &v2) {
  const int dimension = v1.size();
  
  float distance = 0.0;
  for (int i=0; i<dimension; i++) {
    distance += abs(v1[i]-v2[i]);
  }
  
  return distance;
}

#endif