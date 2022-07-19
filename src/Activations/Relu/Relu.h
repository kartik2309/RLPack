//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_RELU_H
#define RLPACK_RELU_H

#include <torch/nn.h>
#include "../../utils/Base/ActivationBase/ActivationBase.h"

class Relu : public ActivationBase {
public:
    torch::nn::ReLU activation_;

    Relu();

    explicit Relu(torch::nn::ReLUOptions &reluOptions);

    torch::Tensor operator()(torch::Tensor &tensor) override;

};


#endif //RLPACK_RELU_H
