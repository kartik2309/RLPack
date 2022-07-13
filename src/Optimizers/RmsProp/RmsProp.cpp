//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "RmsProp.h"

namespace optimizer {
    RmsProp::RmsProp(const std::vector<torch::Tensor> &parameters,
                     const std::shared_ptr<torch::optim::RMSpropOptions> &rmsPropOptions
    ) : OptimizerBase(parameters, rmsPropOptions) {
        optim = std::make_shared<torch::optim::RMSprop>(parameters, *rmsPropOptions);
    }

    torch::Tensor RmsProp::step(torch::optim::Optimizer::LossClosure closure) {
        return optim->step(closure);
    }
}
