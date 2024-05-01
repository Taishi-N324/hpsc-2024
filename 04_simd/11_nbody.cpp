#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
    const int N = 8;
    float x[N], y[N], m[N], fx[N], fy[N];
    for(int i=0; i<N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }

    __m512 xvec = _mm512_load_ps(x);
    __m512 yvec = _mm512_load_ps(y);
    __m512 mvec = _mm512_load_ps(m);

    for(int i=0; i<N; i++) {
        __m512 vxi = _mm512_set1_ps(x[i]);
        __m512 vyi = _mm512_set1_ps(y[i]);

        __m512 rxvec = _mm512_sub_ps(vxi, xvec);
        __m512 ryvec = _mm512_sub_ps(vyi, yvec);

        __m512 r2vec = _mm512_fmadd_ps(rxvec, rxvec, _mm512_mul_ps(ryvec, ryvec));
        __m512 rvec = _mm512_rsqrt14_ps(r2vec); 
        __m512 r3vec = _mm512_mul_ps(_mm512_mul_ps(rvec, rvec), rvec); 

        __mmask16 mask = _mm512_int2mask(~(1 << i));
        __m512 fxvec = _mm512_maskz_mul_ps(mask, rxvec, _mm512_mul_ps(mvec, r3vec));
        __m512 fyvec = _mm512_maskz_mul_ps(mask, ryvec, _mm512_mul_ps(mvec, r3vec));

        fx[i] -= _mm512_reduce_add_ps(fxvec);
        fy[i] -= _mm512_reduce_add_ps(fyvec);
        printf("%d %g %g\n", i, fx[i], fy[i]);
    }

    return 0;
}
