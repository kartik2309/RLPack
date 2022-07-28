//
// Created by Kartik Rajeshwaran on 2022-07-01.
//

#include "Dlqn1d.h"

namespace model::dqn {
    dqn::Dlqn1d::Dlqn1d(
            int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
            std::optional<std::shared_ptr<ActivationBase>> activation, float_t dropout
    ) : model::ModelBase() {

        sequenceLength_ = sequenceLength;
        hiddenSizes_ = hiddenSizes;
        hiddenSizes_.push_back(numAction);

        dropout_ = dropout;
        activationSubmodule = activation.has_value() ? activation.value() : std::make_shared<Relu>();
        numBlocks_ = hiddenSizes_.size() - 1;

        setup_model();
    }

    Dlqn1d::Dlqn1d(std::unique_ptr<Dlqn1dOptions> &options) : model::ModelBase() {
        sequenceLength_ = options->get_sequence_length();
        hiddenSizes_ = options->get_hidden_sizes();
        hiddenSizes_.push_back(options->get_num_actions());

        dropout_ = options->get_dropout();
        activationSubmodule = options->get_activation().has_value()
                              ? options->get_activation().value()
                              : std::make_shared<Relu>();

        numBlocks_ = hiddenSizes_.size() - 1;
        setup_model();
    }

    torch::Tensor dqn::Dlqn1d::forward(torch::Tensor x) {
        for (int32_t idx = 0; idx != numBlocks_; idx++) {

            if (idx == numBlocks_ - 1) {
                x = flattenSubmodule->ptr()->forward(x);
                x = dropoutSubmodule->ptr()->forward(x);
            }

            x = linearSubmodules[idx]->template as<torch::nn::Linear>()->forward(x);

            if (idx != numBlocks_ - 1) {
                x = activationSubmodule->operator()(x);
            }
        }
        return x;
    }

    void dqn::Dlqn1d::setup_model() {

        std::string linearModuleName = "linear_";
        for (int32_t idx = 0; idx != numBlocks_; idx++) {

            if (idx == numBlocks_ - 1) {
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

    dqn::Dlqn1d::~Dlqn1d() = default;
}//namespace model::dqn
