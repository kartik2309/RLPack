//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Relu.h"

torch::Tensor Relu::operator()(torch::Tensor &tensor) {
    return relu_(tensor);
}

Relu::Relu(torch::nn::ReLUOptions &reluOptions) {
    relu_ = torch::nn::ReLU(reluOptions);
}

Relu::Relu() {
    relu_ = torch::nn::ReLU();
}
