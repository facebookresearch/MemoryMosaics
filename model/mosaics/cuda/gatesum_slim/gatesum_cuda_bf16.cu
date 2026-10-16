// Copyright (c) Meta Platforms, Inc. and affiliates.
// See file LICENSE.txt in the main directory.

#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

// GateSum Slim: minimal recurrence CUDA kernel.
//
// Forward:  y_t = omega_t * y_{t-1} + u_t
// Backward: single pass, no register arrays, no exp(), no branching.
//
// omega: (B, T, C_wk) bf16 — decay factor (compact, grouped)
// u:     (B, T, C)    bf16 — gated value (full channels)
// y:     (B, T, C)    bf16 — raw state output
//
// G = C / C_wk channels share one omega value.

__device__ __forceinline__ float warp_reduce_sum(float val, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset, width);
    }
    return val;
}


// ─────────────────────────────────────────────────────────────
// Forward
// ─────────────────────────────────────────────────────────────
__global__ void kernel_forward_slim(const int B, const int T, const int C, const int C_wk,
                                    const bf16 *__restrict__ const _omega,
                                    const bf16 *__restrict__ const _u,
                                    bf16 *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int G = C / C_wk;
    const int _c_wk = _c / G;
    const int _offset_u = _b * T * C + _c;
    const int _offset_omega = _b * T * C_wk + _c_wk;

    const bf16 *__restrict__ const omega = _omega + _offset_omega;
    const bf16 *__restrict__ const u = _u + _offset_u;
    bf16 *__restrict__ const y = _y + _offset_u;

    float y_prev = 0;

    for (int t = 0; t < T; t++) {
        const int ii_u = t * C;
        const int ii_omega = t * C_wk;
        const float omega_val = float(omega[ii_omega]);
        const float u_val = float(u[ii_u]);
        const float y_cur = omega_val * y_prev + u_val;

        y[ii_u] = bf16(y_cur);
        y_prev = y_cur;
    }
}


// ─────────────────────────────────────────────────────────────
// Forward with cache (inference, chunked generation)
// ─────────────────────────────────────────────────────────────
__global__ void kernel_forward_cache_slim(const int B, const int T, const int C, const int C_wk,
                                          const bf16 *__restrict__ const _omega,
                                          const bf16 *__restrict__ const _u,
                                          bf16 *__restrict__ const _y,
                                          float *__restrict__ const _y_prev) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int G = C / C_wk;
    const int _c_wk = _c / G;
    const int _offset_u = _b * T * C + _c;
    const int _offset_omega = _b * T * C_wk + _c_wk;
    const int _prev_offset = _b * C + _c;

    const bf16 *__restrict__ const omega = _omega + _offset_omega;
    const bf16 *__restrict__ const u = _u + _offset_u;
    bf16 *__restrict__ const y = _y + _offset_u;

    float y_prev = _y_prev[_prev_offset];

    for (int t = 0; t < T; t++) {
        const int ii_u = t * C;
        const int ii_omega = t * C_wk;
        const float omega_val = float(omega[ii_omega]);
        const float u_val = float(u[ii_u]);
        const float y_cur = omega_val * y_prev + u_val;

        y[ii_u] = bf16(y_cur);
        y_prev = y_cur;
    }
    _y_prev[_prev_offset] = y_prev;
}


// ─────────────────────────────────────────────────────────────
// Forward with state pad (prefill, variable-length)
// ─────────────────────────────────────────────────────────────
__global__ void kernel_forward_state_pad_slim(const int B, const int T, const int C, const int C_wk,
                                              const bf16 *__restrict__ const _omega,
                                              const bf16 *__restrict__ const _u,
                                              bf16 *__restrict__ const _y,
                                              const int *__restrict__ const _loc,
                                              float *__restrict__ const _y_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int G = C / C_wk;
    const int _c_wk = _c / G;
    const int _offset_u = _b * T * C + _c;
    const int _offset_omega = _b * T * C_wk + _c_wk;
    const int _offset_state = _b * C + _c;

    const bf16 *__restrict__ const omega = _omega + _offset_omega;
    const bf16 *__restrict__ const u = _u + _offset_u;
    bf16 *__restrict__ const y = _y + _offset_u;
    float *__restrict__ const y_state = _y_state + _offset_state;
    const int *__restrict__ const loc = _loc + _b;

    float y_prev = 0;
    int loc_curB = loc[0];

    for (int t = 0; t < T; t++) {
        const int ii_u = t * C;
        const int ii_omega = t * C_wk;
        const float omega_val = float(omega[ii_omega]);
        const float u_val = float(u[ii_u]);
        const float y_cur = omega_val * y_prev + u_val;

        y[ii_u] = bf16(y_cur);
        y_prev = y_cur;

        if (t == loc_curB) {
            y_state[0] = y_cur;
            break;
        }
    }
}


// ─────────────────────────────────────────────────────────────
// Backward: single pass, no register arrays
// ─────────────────────────────────────────────────────────────
__global__ void kernel_backward_slim(const int B, const int T, const int C, const int C_wk,
                                     const bf16 *__restrict__ const _omega,
                                     const bf16 *__restrict__ const _y_saved,
                                     const bf16 *__restrict__ const _gy,
                                     float *__restrict__ const _gomega,
                                     bf16 *__restrict__ const _gu) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int G = C / C_wk;
    const int _c_wk = _c / G;
    const int _offset_u = _b * T * C + _c;
    const int _offset_omega = _b * T * C_wk + _c_wk;

    const bf16 *__restrict__ const omega = _omega + _offset_omega;
    const bf16 *__restrict__ const y_saved = _y_saved + _offset_u;
    const bf16 *__restrict__ const gy = _gy + _offset_u;
    float *__restrict__ const gomega = _gomega + _offset_omega;
    bf16 *__restrict__ const gu = _gu + _offset_u;

    const int group_width = (G <= 32) ? G : 32;
    const int lane = _c % group_width;

    float h = float(gy[(T - 1) * C]);

    for (int t = T - 1; t >= 0; t--) {
        const int ii_u = t * C;
        const int ii_omega = t * C_wk;
        const float omega_val = float(omega[ii_omega]);

        // grad_u[t] = h
        gu[ii_u] = bf16(h);

        // grad_omega[t] = sum_G(h * y_{t-1}), warp reduced
        const float y_prev = (t > 0) ? float(y_saved[(t - 1) * C]) : 0.0f;
        {
            float gomega_val = warp_reduce_sum(h * y_prev, group_width);
            if (lane == 0) {
                if (G <= 32)
                    gomega[ii_omega] = gomega_val;
                else
                    atomicAdd(&gomega[ii_omega], gomega_val);
            }
        }

        // h update
        if (t > 0) {
            h = omega_val * h + float(gy[(t - 1) * C]);
        }
    }
}


// ─────────────────────────────────────────────────────────────
// Host wrappers
// ─────────────────────────────────────────────────────────────
void cuda_forward_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *u, bf16 *y) {
    dim3 threadsPerBlock(min(C, 128));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_slim<<<numBlocks, threadsPerBlock>>>(B, T, C, C_wk, omega, u, y);
}

void cuda_forward_cache_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *u, bf16 *y, float *y_prev) {
    dim3 threadsPerBlock(min(C, 128));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_cache_slim<<<numBlocks, threadsPerBlock>>>(B, T, C, C_wk, omega, u, y, y_prev);
}

void cuda_forward_state_pad_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *u, bf16 *y, int *loc, float *y_state) {
    dim3 threadsPerBlock(min(C, 128));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_state_pad_slim<<<numBlocks, threadsPerBlock>>>(B, T, C, C_wk, omega, u, y, loc, y_state);
}

void cuda_backward_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *y_saved, bf16 *gy, float *gomega, bf16 *gu) {
    dim3 threadsPerBlock(min(C, 128));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward_slim<<<numBlocks, threadsPerBlock>>>(B, T, C, C_wk, omega, y_saved, gy, gomega, gu);
}
