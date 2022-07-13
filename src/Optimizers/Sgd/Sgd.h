//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_SGD_H
#define RLPACK_SGD_H


#include "../OptimizerBase.h"
#include <torch/optim.h>

namespace optimizer {
    class Sgd : public OptimizerBase {
    public:
        std::shared_ptr<torch::optim::SGD> optim;

        Sgd(
                const std::vector<torch::Tensor> &parameters,
                const std::shared_ptr<torch::optim::SGDOptions> &sgdOptions
        );

        torch::Tensor step(torch::optim::Optimizer::LossClosure closure) override;
    };
}

#endif //RLPACK_SGD_H
