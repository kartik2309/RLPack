//
// Created by Kartik Rajeshwaran on 2022-07-05.
//

#include "Normalization.h"

Normalization::Normalization(int32_t applyNorm) {
    applyNorm_ = applyNorm;
};

Normalization::~Normalization() = default;

torch::Tensor Normalization::apply_normalization(torch::Tensor &tensor, float_t eps, int32_t p, int32_t dim) {
    switch (applyNorm_) {
        case -1:
            return tensor;
        case 0:
            return min_max_normalization(tensor, eps, dim);
        case 1:
            return standardization(tensor, eps, dim);
        case 2:
            return p_normalization(tensor, eps, p, dim);
        default:
            throw std::runtime_error("Invalid value of applyNorm was encountered!");
    }
}

torch::Tensor Normalization::min_max_normalization(torch::Tensor &tensor, float_t eps, int32_t dim) {
    std::tuple<torch::Tensor, torch::Tensor> minTensorTuple = tensor.min(dim, true);
    std::tuple<torch::Tensor, torch::Tensor> maxTensorTuple = tensor.max(dim, true);

    torch::Tensor minTensor = std::get<0>(minTensorTuple);
    torch::Tensor maxTensor = std::get<0>(maxTensorTuple);

    tensor = (tensor - minTensor) / (maxTensor - minTensor + eps);
    return tensor;
}

torch::Tensor Normalization::standardization(torch::Tensor &tensor, float_t eps, int32_t dim) {
    torch::Tensor mean_ = tensor.mean(dim);
    torch::Tensor std_ = tensor.std(dim);

    tensor = (tensor - mean_)/(std_ + eps);
    return tensor;
}

torch::Tensor Normalization::p_normalization(torch::Tensor &tensor, float_t eps, int32_t p, int32_t dim) {
    torch::nn::functional::NormalizeFuncOptions normalizeFuncOptions = torch::nn::functional::NormalizeFuncOptions();
    normalizeFuncOptions.eps(eps);
    normalizeFuncOptions.p(p);
    normalizeFuncOptions.dim(dim);
    tensor = torch::nn::functional::normalize(tensor, normalizeFuncOptions);
    return tensor;
}
