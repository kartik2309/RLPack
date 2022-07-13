//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_LRSCHEDULERBASE_H
#define RLPACK_LRSCHEDULERBASE_H

namespace optimizer::lrScheduler {
    class LrSchedulerBase {
    public:
        explicit LrSchedulerBase();

        ~LrSchedulerBase();

        virtual void step() = 0;

        virtual void *get();
    };
}

#endif //RLPACK_LRSCHEDULERBASE_H
