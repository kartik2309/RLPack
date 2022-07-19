//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_STEPLR_H
#define RLPACK_STEPLR_H

#include <torch/optim.h>
#include "../../utils/Base/LrSchedulerBase/LrSchedulerBase.h"
#include "../../Optimizers/Optimizer.hpp"
#include "StepLrOptions/StepLrOptions.h"


namespace optimizer::lrScheduler {
    class StepLr : public optimizer::lrScheduler::LrSchedulerBase {
    public:
        std::shared_ptr<torch::optim::StepLR> stepLr;

        StepLr(std::shared_ptr<OptimizerBase> &optimizer, uint32_t stepSize, float_t gamma = 0.1);

        explicit StepLr(std::shared_ptr<StepLrOptions> &stepLrOptions);

        void step() override;

        void *get() override;

    private:
        std::shared_ptr<OptimizerBase> optimizer_;
        uint32_t stepSize_;
        float_t gamma_;
        int32_t stepCounter_ = 0;

        void perform_decay() override;

    };
}


#endif //RLPACK_STEPLR_H
