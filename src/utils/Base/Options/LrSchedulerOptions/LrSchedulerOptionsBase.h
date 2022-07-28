//
// Created by Kartik Rajeshwaran on 2022-07-18.
//

#ifndef RLPACK_LRSCHEDULEROPTIONSBASE_H
#define RLPACK_LRSCHEDULEROPTIONSBASE_H

#include "../../../../Optimizers/Optimizer.hpp"

namespace optimizer::lrScheduler {
    class LrSchedulerOptionsBase {
    public:
        explicit LrSchedulerOptionsBase(std::shared_ptr<optimizer::OptimizerBase> &optimizer);

        LrSchedulerOptionsBase();

        ~LrSchedulerOptionsBase();

        void optimizer(std::shared_ptr<optimizer::OptimizerBase> &optimizer);

        [[nodiscard]] std::shared_ptr<optimizer::OptimizerBase> get_optimizer() const;

    protected:
        std::shared_ptr<optimizer::OptimizerBase> optim_;
    };
}


#endif //RLPACK_LRSCHEDULEROPTIONSBASE_H
