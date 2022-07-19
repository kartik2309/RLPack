//
// Created by Kartik Rajeshwaran on 2022-07-18.
//

#include "LrSchedulerOptionsBase.h"

namespace optimizer::lrScheduler {

    LrSchedulerOptionsBase::LrSchedulerOptionsBase(std::shared_ptr<optimizer::OptimizerBase> &optimizer) {
        optim_ = optimizer;
    }

    LrSchedulerOptionsBase::LrSchedulerOptionsBase() = default;

    void LrSchedulerOptionsBase::optimizer(std::shared_ptr<optimizer::OptimizerBase> &optimizer) {
        optim_ = optimizer;
    }

    std::shared_ptr<optimizer::OptimizerBase> LrSchedulerOptionsBase::get_optimizer() const {
        return optim_;
    }

    LrSchedulerOptionsBase::~LrSchedulerOptionsBase() = default;
}