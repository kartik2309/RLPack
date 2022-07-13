//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_AGENT_H_
#define RLPACK_DQN_AGENT_H_

#include <random>
#include <torch/torch.h>
#include <boost/log/trivial.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

#include "../../AgentBase.h"
#include "../../utils/utils.hpp"
#include "../../Optimizers/Optimizer.hpp"
#include "../../LrSchedulers/LrScheduler.hpp"
#include "../../ModelBase.h"
#include "DqnAgentOptions/DqnAgentOptions.h"

namespace agent::dqn {

    class Agent : public agent::AgentBase {


    public:
        Agent(
                std::shared_ptr<model::ModelBase> &targetModel,
                std::shared_ptr<model::ModelBase> &policyModel,
                std::shared_ptr<optimizer::OptimizerBase> &optimizer,
                optimizer::lrScheduler::LrSchedulerBase &lrScheduler, float_t gamma, float_t epsilon,
                float_t minEpsilon, float_t epsilonDecayRate, int32_t epsilonDecayFrequency,
                int32_t memoryBufferSize, int32_t targetModelUpdateRate,
                int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
                std::string &savePath, float_t tau, int32_t applyNorm, int32_t applyNormTo,
                float_t epsForNorm, int32_t pForNorm, int32_t dimForNorm
        );

         explicit Agent(std::unique_ptr<DqnAgentOptions> &dqnAgentOptions);

        ~Agent();

        int32_t
        train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done) override;

        int32_t policy(torch::Tensor &stateCurrent) override;

        void save() override;

        void load() override;

    private:
        std::shared_ptr<model::ModelBase> targetModel_;
        std::shared_ptr<model::ModelBase> policyModel_;
        std::shared_ptr<optimizer::OptimizerBase> optimizer_;
        std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> lrScheduler_;
        float_t gamma_;
        float_t epsilon_;
        float_t minEpsilon_;
        float_t epsilonDecayRate_;
        int32_t epsilonDecayFrequency_;
        int32_t memoryBufferSize_;
        int32_t targetModelUpdateRate_;
        int32_t policyModelUpdateRate_;
        int32_t batchSize_;
        int32_t numActions_;
        std::string savePath_;
        float_t tau_;
        std::shared_ptr<Normalization> normalization_;
        int32_t applyNormTo_;
        float_t epsForNorm_;
        int32_t pForNorm_;
        int32_t dimForNorm_;

        Memory memoryBuffer;

        int targetModelUpdateCounter = 0;
        int policyModelUpdateCounter = 0;
        int epsilonDecayCounter = 0;

        torch::nn::HuberLoss huberLoss_;

        void train_policy_model();

        void update_target_model();

        std::shared_ptr<Memory> load_random_experiences();

        torch::Tensor temporal_difference(torch::Tensor &rewards, torch::Tensor &qValues, torch::Tensor &dones) const;

        void decay_epsilon();

        void clear_memory();

    };

}// namespace dqn
#endif//RLPACK_DQN_AGENT_H_
