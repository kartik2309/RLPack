//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_DQNAGENTOPTIONS_H
#define RLPACK_DQNAGENTOPTIONS_H

#include "../../../utils/Options/AgentOptions/AgentOptionsBase.h"
#include "../../../ModelBase.h"

namespace agent::dqn {
    class DqnAgentOptions : public agent::AgentOptionsBase {
    private:
        std::shared_ptr<model::ModelBase> targetModel_;
        std::shared_ptr<model::ModelBase> policyModel_;

    public:

        DqnAgentOptions();

        DqnAgentOptions(
                std::shared_ptr<model::ModelBase> &targetModel,
                std::shared_ptr<model::ModelBase> &policyModel,
                std::shared_ptr<optimizer::OptimizerBase> &optimizer,
                std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> &lrScheduler,
                float_t gamma, float_t epsilon, float_t minEpsilon, float_t epsilonDecayRate,
                int32_t epsilonDecayFrequency, int32_t memoryBufferSize, int32_t targetModelUpdateRate,
                int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
                std::string &savePath, float_t tau, int32_t applyNorm, int32_t applyNormTo,
                float_t epsForNorm, int32_t pForNorm, int32_t dimForNorm
        );

        ~DqnAgentOptions();

        void set_target_model(std::shared_ptr<model::ModelBase> &targetModel);

        void set_policy_model(std::shared_ptr<model::ModelBase> &policyModel);

        std::shared_ptr<model::ModelBase> get_target_model();

        std::shared_ptr<model::ModelBase> get_policy_model();

    };
}

#endif //RLPACK_DQNAGENTOPTIONS_H
