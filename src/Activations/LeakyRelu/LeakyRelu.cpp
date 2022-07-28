//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "LeakyRelu.h"

LeakyRelu::LeakyRelu() {
    activation_ = torch::nn::LeakyReLU();
}

LeakyRelu::LeakyRelu(torch::nn::LeakyReLUOptions &leakyReluOptions) {
    activation_ = torch::nn::LeakyReLU(leakyReluOptions);
}

torch::Tensor LeakyRelu::operator()(torch::Tensor &tensor) {
    return activation_(tensor);
}
