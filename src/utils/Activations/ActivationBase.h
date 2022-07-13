//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_ACTIVATIONBASE_H
#define RLPACK_ACTIVATIONBASE_H

#include <torch/nn.h>

class ActivationBase {
public:
    ActivationBase();

    ~ActivationBase();

    virtual torch::Tensor operator()(torch::Tensor &tensor);
};

#endif //RLPACK_ACTIVATIONBASE_H
