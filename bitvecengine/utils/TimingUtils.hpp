#ifndef TIMING_UTILS_H_
#define TIMING_UTILS_H_

#include <chrono>

using cputime_t = std::chrono::_V2::steady_clock::time_point;

static inline cputime_t timeNow() {
	return std::chrono::steady_clock::now();
}

static inline int64_t durationNs(cputime_t start, cputime_t end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static inline int64_t durationUs(cputime_t start, cputime_t end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static inline double durationMs(cputime_t start, cputime_t end) {
  return ((double)durationUs(start, end)) / 1000.0;
}

static inline double durationS(cputime_t start, cputime_t end) {
  return ((double)durationUs(start, end)) / 1000000.0;
}

#define START_TIMING(s) cputime_t startTime_##s = timeNow(), endTime_##s
#define END_TIMING_NM(s) endTime_##s = timeNow(); std::cout << "Time: " << durationS(startTime_##s, endTime_##s) << " s" << std::endl;
#define END_TIMING_NM_V(s, v) endTime_##s = timeNow(); if(v) std::cout << "Time: " << durationS(startTime_##s, endTime_##s) << " s" << std::endl;
#define END_TIMING(s, msg) endTime_##s = timeNow(); std::cout << msg << durationS(startTime_##s, endTime_##s) << " s" << std::endl;
#define END_TIMING_V(s, msg, v) endTime_##s = timeNow(); if (v) std::cout << msg << durationS(startTime_##s, endTime_##s) << " s" << std::endl;
#define END_TIMING_RET(s, var) endTime_##s = timeNow(); var = durationS(startTime_##s, endTime_##s);

#endif