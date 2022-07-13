//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_RMSPROP_H
#define RLPACK_RMSPROP_H

#include "../OptimizerBase.h"
#include <torch/optim.h>

namespace optimizer {
    class RmsProp : public OptimizerBase {
    public:
        std::shared_ptr<torch::optim::RMSprop> optim;

        RmsProp(
                const std::vector<torch::Tensor> &parameters,
                const std::shared_ptr<torch::optim::RMSpropOptions> &rmsPropOptions
        );

        torch::Tensor step(torch::optim::Optimizer::LossClosure closure) override;
    };
}


#endif //RLPACK_RMSPROP_H
