//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_MODELOPTIONSBASE_H
#define RLPACK_MODELOPTIONSBASE_H

#include <torch/nn.h>
#include "../../Activations/Activations.hpp"

namespace model {

    class ModelOptionsBase {
    protected:
        std::optional<std::shared_ptr<ActivationBase>> activation_;
        int32_t numAction_{};
    public:

        ModelOptionsBase();

        ModelOptionsBase(int32_t numActions, std::optional<std::shared_ptr<ActivationBase>> &activation);

        ~ModelOptionsBase();

        void activation(std::optional<std::shared_ptr<ActivationBase>> &activation);

        void num_actions(int32_t numActions);

        std::optional<std::shared_ptr<ActivationBase>> get_activation();

        [[nodiscard]] int32_t get_num_actions() const;
    };
}

#endif //RLPACK_MODELOPTIONSBASE_H
