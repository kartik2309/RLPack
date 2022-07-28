//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_SGD_H
#define RLPACK_SGD_H


#include "../../utils/Base/OptimizerBase/OptimizerBase.h"
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

        float get_lr(int paramGroupIndex) override;

        void set_lr(std::vector<float> &newLrVector) const override;

        uint32_t get_param_group_size() override;
    };
}

#endif //RLPACK_SGD_H
