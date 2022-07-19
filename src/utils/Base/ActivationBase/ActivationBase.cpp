//
// Created by Kartik Rajeshwaran on 2022-07-12.
//

#include "ActivationBase.h"

torch::Tensor ActivationBase::operator()(torch::Tensor &tensor) {
    return tensor;
}

ActivationBase::ActivationBase() = default;

ActivationBase::~ActivationBase() = default;