//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "ModelOptionsBase.h"

namespace model {

    ModelOptionsBase::ModelOptionsBase() = default;

    ModelOptionsBase::ModelOptionsBase(int32_t numActions, std::optional<std::shared_ptr<ActivationBase>> &activation) {
        numAction_ = numActions;
        activation_ = std::move(activation);

    }

    void ModelOptionsBase::activation(std::optional<std::shared_ptr<ActivationBase>> &activation) {
        activation_ = activation;
    }

    void ModelOptionsBase::num_actions(int32_t numActions) {
        numAction_ = numActions;
    }

    std::optional<std::shared_ptr<ActivationBase>> ModelOptionsBase::get_activation() {
        return activation_;
    }

    int32_t ModelOptionsBase::get_num_actions() const {
        return numAction_;
    }

    ModelOptionsBase::~ModelOptionsBase() = default;
}

