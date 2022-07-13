//
// Created by Kartik Rajeshwaran on 2022-07-12.
//

#include "LrSchedulerBase.h"

namespace optimizer::lrScheduler {

    void *LrSchedulerBase::get() {
        return nullptr;
    }

    LrSchedulerBase::LrSchedulerBase() = default;

    LrSchedulerBase::~LrSchedulerBase() = default;
}