//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_AGENT_H_
#define RLPACK_DQN_AGENT_H_

#include <random>
#include <torch/torch.h>
#include <boost/log/trivial.hpp>

#include "../AgentBase.hpp"


namespace dqn {
    template<class ModelClass, class Optimizer>
    class Agent : public AgentBase {


    public:
        Agent(ModelClass &targetModel, ModelClass &policyModel, Optimizer &optimizer, float_t gamma,
              float_t epsilon, float_t epsilonDecayRate, int32_t memoryBufferSize, int32_t targetModelUpdateRate,
              int32_t policyModelUpdateRate, int32_t numActions, std::string &savePath, int32_t applyNorm = -1,
              int32_t applyNormTo = 0);

        ~Agent();

        int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done) override;

        int policy(torch::Tensor &stateCurrent) override;

        void save() override;

    private:

        ModelClass targetModel_;
        ModelClass policyModel_;
        Optimizer optimizer_;
        float gamma_;
        float epsilon_;
        float epsilonDecayRate_;
        int memoryBufferSize_;
        int targetModelUpdateRate_;
        int policyModelUpdateRate_;
        int numActions_;
        std::string savePath_;
        int32_t applyNorm_;
        int32_t applyNormTo_;

        torch::nn::HuberLoss huberLoss_;

        struct Memory {
        private:
            std::vector<torch::Tensor> stateCurrent_;
            std::vector<torch::Tensor> stateNext_;
            std::vector<torch::Tensor> reward_;
            std::vector<torch::Tensor> action_;
            std::vector<torch::Tensor> done_;

        public:
            Memory();

            ~Memory();

            void push_back(torch::Tensor &stateCurrent,
                           torch::Tensor &stateNext,
                           float reward,
                           int action,
                           int done);

            void push_back(Memory *memory);

            torch::Tensor stack_current_states(int32_t applyNorm, int32_t applyNormTo);

            torch::Tensor stack_next_states(int32_t applyNorm, int32_t applyNormTo);

            torch::Tensor stack_rewards(int32_t applyNorm, int32_t applyNormTo);

            torch::Tensor stack_actions();

            torch::Tensor stack_dones();

            void clear();

            size_t size();

            Memory *at(int index);
        } memoryBuffer;

        int targetModelUpdateCounter = 0;
        int policyModelUpdateCounter = 0;

        void train_policy_model();

        void update_target_model();

        typename Agent<ModelClass, Optimizer>::Memory *load_random_experiences();

        torch::Tensor temporal_difference(torch::Tensor &rewards, torch::Tensor &qValues, torch::Tensor &dones);

        static torch::Tensor apply_normalization(torch::Tensor &tensor, int32_t applyNorm);

        void decay_epsilon();

        void clear_memory();

    };

}// namespace dqn
#endif//RLPACK_DQN_AGENT_H_
