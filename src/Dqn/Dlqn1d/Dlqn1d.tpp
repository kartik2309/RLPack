//
// Created by Kartik Rajeshwaran on 2022-07-01.
//

#ifndef RLPACK_DQN_DLQN1D_DLQN1D_TPP
#define RLPACK_DQN_DLQN1D_DLQN1D_TPP


#include "Dlqn1d.h"

namespace dqn {
    template<class Activation>
    dqn::Dlqn1d<Activation>::Dlqn1d(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                                    std::optional<Activation> activation, float_t dropout) {

        sequenceLength_ = sequenceLength;
        hiddenSizes_ = hiddenSizes;
        hiddenSizes_.push_back(numAction);

        dropout_ = dropout;
        activationSubmodule = activation.has_value() ? activation.value() : Activation();
        numBlocks_ = hiddenSizes_.size() - 1;

        setup_model();
    }

    template<class Activation>
    torch::Tensor dqn::Dlqn1d<Activation>::forward(torch::Tensor x) {
        for (int32_t idx = 0; idx != numBlocks_; idx++) {

            if (idx == numBlocks_ - 1) {
                x = dropoutSubmodule->ptr()->forward(x);
            }

            x = linearSubmodules[idx]->template as<torch::nn::Linear>()->forward(x);

            if (idx != numBlocks_ - 1) {
                x = activationSubmodule(x);
            }
        }
        return x;
    }

    template<class Activation>
    void dqn::Dlqn1d<Activation>::setup_model() {

        std::string linearModuleName = "linear_";
        for (int32_t idx = 0; idx != numBlocks_; idx++) {

            if (idx == numBlocks_ - 1){
                hiddenSizes_[idx] *= sequenceLength_;
            }

            auto linearOptionsForSubmodule = torch::nn::LinearOptions(hiddenSizes_[idx],
                                                                      hiddenSizes_[idx + 1]);

            auto linearModule = std::make_shared<torch::nn::Linear>(linearOptionsForSubmodule);
            linearSubmodules->push_back(*linearModule);
            register_module(linearModuleName.append(std::to_string(idx)), *linearModule);
        }

        auto dropoutOptions = torch::nn::DropoutOptions(dropout_);
        dropoutSubmodule = std::make_shared<torch::nn::Dropout>(dropoutOptions);
        register_module("dropout", *dropoutSubmodule);

        auto flattenOptions = torch::nn::FlattenOptions().start_dim(1).end_dim(-1);
        flattenSubmodule = std::make_shared<torch::nn::Flatten>(flattenOptions);
        register_module("flatten", *flattenSubmodule);

    }

    template<class Activation>
    dqn::Dlqn1d<Activation>::~Dlqn1d() = default;
}

#endif //RLPACK_DQN_DLQN1D_DLQN1D_TPP