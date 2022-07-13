//
// Created by Kartik Rajeshwaran on 2022-07-12.
//

#include "OptimizerBase.h"

namespace optimizer{

    OptimizerBase::OptimizerBase(
            const std::vector<torch::Tensor> &parameters,
            const std::shared_ptr<torch::optim::OptimizerOptions>& defaults
    ) : torch::optim::Optimizer(parameters, defaults->clone()) {
        optim = nullptr;
    }

    OptimizerBase::~OptimizerBase() = default;
}