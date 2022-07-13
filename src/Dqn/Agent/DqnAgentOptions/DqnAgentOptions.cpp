//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "DqnAgentOptions.h"

namespace agent::dqn {

    DqnAgentOptions::DqnAgentOptions() = default;

    DqnAgentOptions::DqnAgentOptions(
            std::shared_ptr<model::ModelBase> &targetModel,
            std::shared_ptr<model::ModelBase> &policyModel,
            std::shared_ptr<optimizer::OptimizerBase> &optimizer,
            std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> &lrScheduler, float_t gamma,
            float_t epsilon, float_t minEpsilon, float_t epsilonDecayRate,
            int32_t epsilonDecayFrequency, int32_t memoryBufferSize, int32_t targetModelUpdateRate,
            int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
            std::string &savePath, float_t tau, int32_t applyNorm, int32_t applyNormTo,
            float_t epsForNorm, int32_t pForNorm, int32_t dimForNorm
    ) : AgentOptionsBase(
            optimizer, lrScheduler, gamma, epsilon, minEpsilon, epsilonDecayRate,
            epsilonDecayFrequency, memoryBufferSize, targetModelUpdateRate, policyModelUpdateRate, batchSize,
            numActions, savePath, tau, applyNorm, applyNormTo, epsForNorm, pForNorm, dimForNorm
    ) {
        targetModel_ = targetModel;
        policyModel_ = policyModel;
        optimizer_ = optimizer;

        gamma_ = gamma;
        epsilon_ = epsilon;
        minEpsilon_ = minEpsilon;
        epsilonDecayRate_ = epsilonDecayRate;
        epsilonDecayFrequency_ = epsilonDecayFrequency;
        memoryBufferSize_ = memoryBufferSize;
        assert(targetModelUpdateRate > policyModelUpdateRate);

        targetModelUpdateRate_ = targetModelUpdateRate;
        policyModelUpdateRate_ = policyModelUpdateRate;
        batchSize_ = batchSize;
        numActions_ = numActions;

        savePath_ = savePath;

        if (tau < 0 || tau > 1) {
            throw std::range_error("Invalid value for tau passed! Expected value is between 0 and 1");
        }
        tau_ = tau;

        if (applyNormTo < -1 || applyNormTo > 4) {
            throw std::range_error("Invalid applyNormTo passed!");
        }
        applyNormTo_ = applyNormTo;
        normalization_ = std::make_shared<Normalization>(applyNorm);

        epsForNorm_ = epsForNorm;
        pForNorm_ = pForNorm;
        dimForNorm_ = dimForNorm;
    }

    void DqnAgentOptions::set_target_model(std::shared_ptr<model::ModelBase> &targetModel) {
        targetModel_ = targetModel;
    }

    void DqnAgentOptions::set_policy_model(std::shared_ptr<model::ModelBase> &policyModel) {
        policyModel_ = policyModel;
    }


    std::shared_ptr<model::ModelBase> DqnAgentOptions::get_target_model() {
        return targetModel_;
    }

    std::shared_ptr<model::ModelBase> DqnAgentOptions::get_policy_model() {
        return policyModel_;
    }

    DqnAgentOptions::~DqnAgentOptions() = default;
}

