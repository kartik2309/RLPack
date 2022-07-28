//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_RMSPROP_H
#define RLPACK_RMSPROP_H

#include "../../utils/Base/OptimizerBase/OptimizerBase.h"
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

        float get_lr(int paramGroupIndex) override;

        void set_lr(std::vector<float> &newLrVector) const override;

        uint32_t get_param_group_size() override;
    };
}


#endif //RLPACK_RMSPROP_H
