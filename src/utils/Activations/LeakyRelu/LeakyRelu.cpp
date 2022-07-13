//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "LeakyRelu.h"

LeakyRelu::LeakyRelu() {
    leakyRelu_ = torch::nn::LeakyReLU();
}

LeakyRelu::LeakyRelu(torch::nn::LeakyReLUOptions &leakyReluOptions) {
    leakyRelu_ = torch::nn::LeakyReLU(leakyReluOptions);
}

torch::Tensor LeakyRelu::operator()(torch::Tensor &tensor) {
    return leakyRelu_(tensor);
}
