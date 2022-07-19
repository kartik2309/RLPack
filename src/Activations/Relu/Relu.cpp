//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Relu.h"

torch::Tensor Relu::operator()(torch::Tensor &tensor) {
    return activation_(tensor);
}

Relu::Relu(torch::nn::ReLUOptions &reluOptions) {
    activation_ = torch::nn::ReLU(reluOptions);
}

Relu::Relu() {
    activation_ = torch::nn::ReLU();
}
