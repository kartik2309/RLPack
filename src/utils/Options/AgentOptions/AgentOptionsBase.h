//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_AGENTOPTIONSBASE_H
#define RLPACK_AGENTOPTIONSBASE_H

#include <torch/torch.h>
#include "../../../Optimizers/Optimizer.hpp"
#include "../../../LrSchedulers/LrSchedulerBase.h"
#include "../../Normalization/Normalization.h"

namespace agent {
    class AgentOptionsBase {

    protected:

        std::shared_ptr<optimizer::OptimizerBase> optimizer_;
        std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> lrScheduler_ = nullptr;
        float_t gamma_ = 0.9;
        float_t epsilon_ = 0.5;
        float_t minEpsilon_ = 0.01;
        float_t epsilonDecayRate_ = 0.99;
        int32_t epsilonDecayFrequency_ = 16;
        int32_t memoryBufferSize_ = 16384;
        int32_t targetModelUpdateRate_ = 512;
        int32_t policyModelUpdateRate_ = 8;
        int32_t modelBackFrequency_ = 128;
        int32_t batchSize_ = 64;
        int32_t numActions_ = 4;
        std::string savePath_ = "./";
        float_t tau_ = 1.0;
        std::shared_ptr<Normalization> normalization_ = nullptr;
        int32_t applyNorm_ = -1;
        int32_t applyNormTo_ = -1;
        float_t epsForNorm_ = 5e-8;
        int32_t pForNorm_ = 2;
        int32_t dimForNorm_ = 0;

    public:

        AgentOptionsBase();

        AgentOptionsBase(
                std::shared_ptr<optimizer::OptimizerBase> &optimizer,
                std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> &lrScheduler,
                float_t gamma, float_t epsilon, float_t minEpsilon, float_t epsilonDecayRate,
                int32_t epsilonDecayFrequency, int32_t memoryBufferSize, int32_t targetModelUpdateRate,
                int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
                std::string &savePath, float_t tau, int32_t applyNorm, int32_t applyNormTo,
                float_t epsForNorm, int32_t pForNorm, int32_t dimForNorm
        );

        ~AgentOptionsBase();

        // Setter Methods

        void set_optimizer(std::shared_ptr<optimizer::OptimizerBase> &optimizer);

        void set_lr_scheduler(std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> &lrScheduler);

        void gamma(float_t gamma);

        void epsilon(float_t epsilon);

        void min_epsilon(float_t minEpsilon);

        void epsilon_decay_rate(float_t epsilonDecayRate);

        void epsilon_decay_frequency(int32_t epsilonDecayFrequency);

        void memory_buffer_size(int32_t memoryBufferSize);

        void target_model_update_rate(int32_t targetModelUpdateRate);

        void policy_model_update_rate(int32_t policyModelUpdateRate);

        void model_backup_frequency(int32_t modelBackFrequency);

        void batch_size(int32_t batchSize);

        void num_actions(int32_t numActions);

        void save_path(std::string &savePath);

        void tau(float_t tau = 1);

        void apply_norm(int32_t applyNorm = -1);

        void apply_norm_to(int32_t applyNormTo = -1);

        void eps_for_norm(float_t epsForNorm = 5e-8);

        void p_for_norm(int32_t pForNorm = 2);

        void dim_for_norm(int32_t dimForNorm = 0);

        // Getter Methods

        std::shared_ptr<optimizer::OptimizerBase> get_optimizer();

        std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> get_lr_scheduler();

        [[nodiscard]] float_t get_gamma() const;

        [[nodiscard]] float_t get_epsilon() const;

        [[nodiscard]] float_t get_min_epsilon() const;

        [[nodiscard]] float_t get_epsilon_decay_rate() const;

        [[nodiscard]] int32_t get_epsilon_decay_frequency() const;

        [[nodiscard]] int32_t get_memory_buffer_size() const;

        [[nodiscard]] int32_t get_target_model_update_rate() const;

        [[nodiscard]] int32_t get_policy_model_update_rate() const;

        [[nodiscard]] int32_t get_model_backup_frequency() const;

        [[nodiscard]] int32_t get_batch_size() const;

        [[nodiscard]] int32_t get_num_actions() const;

        std::string get_save_path();

        [[nodiscard]] float_t get_tau() const;

        [[nodiscard]] int32_t get_apply_norm() const;

        [[nodiscard]] int32_t get_apply_norm_to() const;

        [[nodiscard]] float_t get_eps_for_norm() const;

        [[nodiscard]] int32_t get_p_for_norm() const;

        [[nodiscard]] int32_t get_dim_for_norm() const;

        std::shared_ptr<Normalization>get_normalizer();
    };

}
#endif //RLPACK_AGENTOPTIONSBASE_H
