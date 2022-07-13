//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Sgd.h"

namespace optimizer {
    Sgd::Sgd(const std::vector<torch::Tensor> &parameters,
             const std::shared_ptr<torch::optim::SGDOptions> &sgdOptions
    ) : OptimizerBase(parameters, sgdOptions) {
        optim = std::make_shared<torch::optim::SGD>(parameters, *sgdOptions);

    }

    torch::Tensor Sgd::step(torch::optim::Optimizer::LossClosure closure) {
        return optim->step(closure);
    }
}

