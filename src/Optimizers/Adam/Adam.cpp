//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Adam.h"

namespace optimizer {
    Adam::Adam(const std::vector<torch::Tensor> &parameters,
               const std::shared_ptr<torch::optim::AdamOptions>& adamOptions
    ) : OptimizerBase(parameters, adamOptions) {
        std::shared_ptr<torch::optim::AdamOptions> adamOptions_ = std::make_shared<torch::optim::AdamOptions>(std::move(*adamOptions));
        optim = std::make_shared<torch::optim::Adam>(parameters, *adamOptions_);
    }

    torch::Tensor Adam::step(torch::optim::Optimizer::LossClosure closure) {
        return optim->step(closure);
    }
}

