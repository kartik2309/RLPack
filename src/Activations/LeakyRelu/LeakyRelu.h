//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_LEAKYRELU_H
#define RLPACK_LEAKYRELU_H

#include <torch/nn.h>
#include "../../utils/Base/ActivationBase/ActivationBase.h"

class LeakyRelu : public ActivationBase{
public:
    torch::nn::LeakyReLU activation_;

    LeakyRelu();

    explicit LeakyRelu(torch::nn::LeakyReLUOptions &leakyReluOptionsOptions);

    torch::Tensor operator()(torch::Tensor &tensor) override;
};


#endif //RLPACK_LEAKYRELU_H
