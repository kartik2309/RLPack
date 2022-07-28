//
// Created by Kartik Rajeshwaran on 2022-07-18.
//

#ifndef RLPACK_STEPLROPTIONS_H
#define RLPACK_STEPLROPTIONS_H

# include "../../../utils/Base/Options/LrSchedulerOptions/LrSchedulerOptionsBase.h"

namespace optimizer::lrScheduler {
    class StepLrOptions : public LrSchedulerOptionsBase {

    public:
        StepLrOptions();

        StepLrOptions(std::shared_ptr<OptimizerBase> &optimizer, uint32_t stepSize, float_t gamma);

        ~StepLrOptions();

        void step_size(uint32_t stepSize);

        void gamma(float_t gamma);

        [[nodiscard]] uint32_t get_step_size() const;

        [[nodiscard]] float get_gamma() const;

    private:
        uint32_t stepSize_ = 0;
        float_t gamma_ = 1;
    };
}


#endif //RLPACK_STEPLROPTIONS_H
