//
// Created by Kartik Rajeshwaran on 2022-07-18.
//

#include "StepLrOptions.h"

namespace optimizer::lrScheduler {
    optimizer::lrScheduler::StepLrOptions::StepLrOptions() = default;

    optimizer::lrScheduler::StepLrOptions::StepLrOptions(
            std::shared_ptr<OptimizerBase> &optimizer, uint32_t stepSize, float_t gamma) : LrSchedulerOptionsBase() {
        optim_ = optimizer;
        stepSize_ = stepSize;
        gamma_ = gamma;
    }

    void StepLrOptions::step_size(uint32_t stepSize) {
        stepSize_ = stepSize;
    }

    void StepLrOptions::gamma(float_t gamma) {
        gamma_ = gamma;
    }

    uint32_t StepLrOptions::get_step_size() const {
        return stepSize_;
    }

    float StepLrOptions::get_gamma() const {
        return gamma_;
    }

    StepLrOptions::~StepLrOptions() = default;
}