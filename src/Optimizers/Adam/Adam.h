//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_ADAM_H
#define RLPACK_ADAM_H

#include <torch/optim.h>
#include "../../utils/Base/OptimizerBase/OptimizerBase.h"

namespace optimizer {
    class Adam : public OptimizerBase {


    public:
        std::shared_ptr<torch::optim::Adam> optim;

        Adam(const std::vector<torch::Tensor> &parameters,
             const std::shared_ptr<torch::optim::AdamOptions> &adamOptions
        );

        torch::Tensor step(torch::optim::Optimizer::LossClosure closure) override;

        float get_lr(int paramGroupIndex) override;

        void set_lr(std::vector<float> &newLrVector) const override;

        uint32_t get_param_group_size() override;
    };
}


#endif //RLPACK_ADAM_H
