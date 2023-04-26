#ifndef MATH_H_
#define MATH_H_

#include <Eigen/Core>
#include "Types.hpp"


float fvec_L2sqr_ref (const float * x,
                     const float * y,
                     size_t d)
{
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
       res += tmp * tmp;
    }
    return res;
}
float fvec_L2sqr (const float * x,
                  const float * y,
                  size_t d)
{
    return fvec_L2sqr_ref (x, y, d);
}
void fvec_L2sqr_ny_ref (float * dis,
                    const float * x,
                    const float * y,
                    size_t d, size_t ny)
{
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr (x, y, d);
        y += d;
    }
}

#ifdef __AVX2__
template<class ElementOp>
void fvec_op_ny_D1 (float * dis, const float * x,
                       const float * y, size_t ny)
{
    float x0s = x[0];
    __m128 x0 = _mm_set_ps (x0s, x0s, x0s, x0s);

    size_t i;
    for (i = 0; i + 3 < ny; i += 4) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps (y)); y += 4;
        dis[i] = _mm_cvtss_f32 (accu);
        __m128 tmp = _mm_shuffle_ps (accu, accu, 1);
        dis[i + 1] = _mm_cvtss_f32 (tmp);
        tmp = _mm_shuffle_ps (accu, accu, 2);
        dis[i + 2] = _mm_cvtss_f32 (tmp);
        tmp = _mm_shuffle_ps (accu, accu, 3);
        dis[i + 3] = _mm_cvtss_f32 (tmp);
    }
    while (i < ny) { // handle non-multiple-of-4 case
        dis[i++] = ElementOp::op(x0s, *y++);
    }
}

template<class ElementOp>
void fvec_op_ny_D2 (float * dis, const float * x,
                       const float * y, size_t ny)
{
    __m128 x0 = _mm_set_ps (x[1], x[0], x[1], x[0]);

    size_t i;
    for (i = 0; i + 1 < ny; i += 2) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps (y)); y += 4;
        accu = _mm_hadd_ps (accu, accu);
        dis[i] = _mm_cvtss_f32 (accu);
        accu = _mm_shuffle_ps (accu, accu, 3);
        dis[i + 1] = _mm_cvtss_f32 (accu);
    }
    if (i < ny) { // handle odd case
        dis[i] = ElementOp::op(x[0], y[0]) + ElementOp::op(x[1], y[1]);
    }
}



template<class ElementOp>
void fvec_op_ny_D4 (float * dis, const float * x,
                        const float * y, size_t ny)
{
    __m128 x0 = _mm_loadu_ps(x);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps (y)); y += 4;
        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        dis[i] = _mm_cvtss_f32 (accu);
    }
}

template<class ElementOp>
void fvec_op_ny_D8 (float * dis, const float * x,
                        const float * y, size_t ny)
{
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps (y)); y += 4;
        accu       += ElementOp::op(x1, _mm_loadu_ps (y)); y += 4;
        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        dis[i] = _mm_cvtss_f32 (accu);
    }
}

template<class ElementOp>
void fvec_op_ny_D12 (float * dis, const float * x,
                        const float * y, size_t ny)
{
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);
    __m128 x2 = _mm_loadu_ps(x + 8);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps (y)); y += 4;
        accu       += ElementOp::op(x1, _mm_loadu_ps (y)); y += 4;
        accu       += ElementOp::op(x2, _mm_loadu_ps (y)); y += 4;
        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        dis[i] = _mm_cvtss_f32 (accu);
    }
}
#endif

struct ElementOpL2 {

    static float op (float x, float y) {
        float tmp = x - y;
        return tmp * tmp;
    }

#ifdef __AVX2__
    static __m128 op (__m128 x, __m128 y) {
        __m128 tmp = x - y;
        return tmp * tmp;
    }
#endif

};

void fvec_L2sqr_ny (float * dis, const float * x,
                        const float * y, size_t d, size_t ny) {
    // optimized for a few special cases
#ifdef __AVX2__
#define DISPATCH(dval) \
    case dval:\
        fvec_op_ny_D ## dval <ElementOpL2> (dis, x, y, ny); \
        return;

    switch(d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
    default:
        fvec_L2sqr_ny_ref (dis, x, y, d, ny);
        return;
    }
#undef DISPATCH
#else
  fvec_L2sqr_ny_ref (dis, x, y, d, ny);
#endif

}

inline Eigen::VectorXf cumSum(const Eigen::VectorXf &Z){
  Eigen::VectorXf ret(Z.rows());
  ret(0) = Z(0);
  for (int i=1; i<Z.rows(); i++) {
    ret(i) = Z(i) + ret(i-1);
  }

  return ret;
}

inline int nextPow2(double x) {
  if (x == 0)  {
    return 0;
  }
  return (int)std::pow(2, std::floor(std::log2(std::abs(x))));
}

inline RowVector<float> percentile(const LUTType &x, float percent) {
  // percentile for each col
  RowVector<float> ret(x.cols());
  auto xPtr = x.data();
  for (int i=0; i<x.cols(); i++) {
    std::vector<float> temp(x.rows());
    std::copy(xPtr, xPtr + x.rows(), temp.begin());
    std::sort(temp.begin(), temp.end());

    float nthF = percent * static_cast<float>(x.rows() - 1);
    if (std::fabs(std::round(nthF) - nthF) <= 0.00001f) {
      ret(i) = temp.at(static_cast<int>(nthF));
    } else {
      float f = temp.at(static_cast<int>(std::floor(nthF))),
            c = temp.at(static_cast<int>(std::ceil(nthF)));
      float fraction = nthF - std::round(nthF);
      ret(i) = f + (c - f) * fraction;
    }

    xPtr += x.rows();
  }
  
  return ret;
}

inline SmallLUTType smallQuantize(const LUTType &lut_offset, const ColVector<float> &scaleBy) {
  SmallLUTType luts_quantized(lut_offset.rows(), lut_offset.cols());
  for (int i=0; i<lut_offset.cols(); i++) {
    luts_quantized.col(i) = (lut_offset.col(i) * scaleBy(i)).unaryExpr(
      [](float x){return std::min(std::floor(x), 255.0f);}
    ).cast<uint8_t>();
  }

  return luts_quantized;
}

inline ColMatrix<float> max(const ColMatrix<float> &x) {
  ColMatrix<float> ret = x;
  for (int i=0; i<ret.cols(); i++) {
    for (int j=0; j<ret.rows(); j++) {
      if (ret(j, i) < 0) {
        ret(j, i) = 0;
      }
    }
  }

  return ret;
}

float Floor(float x) {
  return std::floor(x);
}

inline RowMatrixXf computeCovarianceMat(const RowMatrixXf &X) {
  RowMatrixXf covmat(X.cols(), X.cols());
  covmat.setZero();
  #pragma omp parallel for num_threads(2) collapse(2)
  for (int r=0; r<covmat.cols(); r++) {
    for (int c=0; c<covmat.cols(); c++) {
      for (int d=0; d<X.rows(); d++) {
        covmat(r, c) += X(d, r) * X(d, c);
      }
    }
  }

  // #pragma omp parallel for num_threads(2)
  // for (int n=0; n<covmat.cols()*covmat.cols(); n++) {
  //   int r = n/covmat.cols(); int c = n%covmat.cols();
  //   for (int d=0; d<X.rows(); d++) {
  //     covmat(r, c) += X(d, r) * X(d, c);
  //   }
  // }

  return covmat;
}

#endif  // MATH_H_