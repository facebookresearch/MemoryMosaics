// Copyright (c) Meta Platforms, Inc. and affiliates.
// See file LICENSE.txt in the main directory.

#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *u, bf16 *y);
void cuda_forward_cache_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *u, bf16 *y, float *y_prev);
void cuda_forward_state_pad_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *u, bf16 *y, int *loc, float *y_state);
void cuda_backward_slim(int B, int T, int C, int C_wk, bf16 *omega, bf16 *y_saved, bf16 *gy, float *gomega, bf16 *gu);

void forward(int64_t B, int64_t T, int64_t C, int64_t C_wk,
             torch::Tensor &omega, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_slim(B, T, C, C_wk,
        omega.data_ptr<bf16>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}

void forward_cache(int64_t B, int64_t T, int64_t C, int64_t C_wk,
                   torch::Tensor &omega, torch::Tensor &u, torch::Tensor &y, torch::Tensor &y_prev) {
    cuda_forward_cache_slim(B, T, C, C_wk,
        omega.data_ptr<bf16>(), u.data_ptr<bf16>(), y.data_ptr<bf16>(), y_prev.data_ptr<float>());
}

void forward_state_pad(int64_t B, int64_t T, int64_t C, int64_t C_wk,
                       torch::Tensor &omega, torch::Tensor &u, torch::Tensor &y,
                       torch::Tensor &loc, torch::Tensor &y_state) {
    cuda_forward_state_pad_slim(B, T, C, C_wk,
        omega.data_ptr<bf16>(), u.data_ptr<bf16>(), y.data_ptr<bf16>(),
        loc.data_ptr<int>(), y_state.data_ptr<float>());
}

void backward(int64_t B, int64_t T, int64_t C, int64_t C_wk,
              torch::Tensor &omega, torch::Tensor &y_saved, torch::Tensor &gy,
              torch::Tensor &gomega, torch::Tensor &gu) {
    cuda_backward_slim(B, T, C, C_wk,
        omega.data_ptr<bf16>(), y_saved.data_ptr<bf16>(), gy.data_ptr<bf16>(),
        gomega.data_ptr<float>(), gu.data_ptr<bf16>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "gatesum_slim forward");
    m.def("forward_cache", &forward_cache, "gatesum_slim forward_cache");
    m.def("forward_state_pad", &forward_state_pad, "gatesum_slim forward_state_pad");
    m.def("backward", &backward, "gatesum_slim backward");
}

TORCH_LIBRARY(gatesum_slim_cuda, m) {
    m.def("forward", forward);
    m.def("forward_cache", forward_cache);
    m.def("forward_state_pad", forward_state_pad);
    m.def("backward", backward);
}
