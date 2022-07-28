//
// Created by Kartik Rajeshwaran on 2022-07-21.
//

#include "Ops.h"

torch::Tensor Ops::vector_to_tensor(std::vector<std::vector<float_t>> &vector) {
    std::vector<float_t> vector_ = vector[0];
    std::vector<float_t> shape_ = vector[1];

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::zeros(std::vector<int64_t>(shape_.begin(), shape_.end()), options);
    std::memmove(tensor.data_ptr<float_t>(), vector_.data(), sizeof(float_t) * tensor.numel());
    return tensor;
}

torch::Tensor Ops::vector_to_tensor(std::vector<std::vector<double_t>> &vector) {
    std::vector<double_t> vector_ = vector[0];
    std::vector<double_t> shape_ = vector[1];

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor tensor = torch::zeros(std::vector<int64_t>(shape_.begin(), shape_.end()), options);
    std::memmove(tensor.data_ptr<double_t>(), vector_.data(), sizeof(double_t) * tensor.numel());
    return tensor;
}
