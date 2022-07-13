//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "StepLr.h"

namespace optimizer::lrScheduler {

    optimizer::lrScheduler::StepLr::StepLr(
            std::shared_ptr<OptimizerBase> &optimizer, const uint32_t stepSize, const float_t gamma) {
        stepLr = std::make_shared<torch::optim::StepLR>(*optimizer->optim, stepSize, gamma);
    }

    void optimizer::lrScheduler::StepLr::step() {
        stepLr->step();
    }

    void *StepLr::get() {
        return stepLr.get();
    }
}
