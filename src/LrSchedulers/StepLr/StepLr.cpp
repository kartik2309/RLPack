//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "StepLr.h"

namespace optimizer::lrScheduler {

    optimizer::lrScheduler::StepLr::StepLr(std::shared_ptr<OptimizerBase> &optimizer, uint32_t stepSize,
                                           float_t gamma) {
        optimizer_ = optimizer;
        stepSize_ = stepSize;
        gamma_ = gamma;
    }

    StepLr::StepLr(std::shared_ptr<StepLrOptions> &stepLrOptions) {
        optimizer_ = stepLrOptions->get_optimizer();
        stepSize_ = stepLrOptions->get_step_size();
        gamma_ = stepLrOptions->get_gamma();
    }

    void optimizer::lrScheduler::StepLr::step() {
        stepCounter_ += 1;

        if (stepCounter_ % stepSize_ == 0) {
            perform_decay();
        }
    }

    void *StepLr::get() {
        return stepLr.get();
    }

    void StepLr::perform_decay() {
        std::vector<float> newLrs;
        uint32_t paramGroupSize = optimizer_->get_param_group_size();

        for (int idx = 0; idx != paramGroupSize; idx++) {
            float lr = optimizer_->get_lr(idx);
            lr *= gamma_;
            newLrs.push_back(lr);
        }

        optimizer_->set_lr(newLrs);
    }

}
