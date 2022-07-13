//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_STEPLR_H
#define RLPACK_STEPLR_H

#include <torch/optim.h>
#include "../LrSchedulerBase.h"
#include "../../Optimizers/Optimizer.hpp"


namespace optimizer::lrScheduler {
    class StepLr : public optimizer::lrScheduler::LrSchedulerBase {
    public:
        std::shared_ptr<torch::optim::StepLR> stepLr;

        StepLr(std::shared_ptr<OptimizerBase> &optimizer, uint32_t stepSize, float_t gamma = 0.1);

        void step() override;

        void *get() override;

    };
}


#endif //RLPACK_STEPLR_H
