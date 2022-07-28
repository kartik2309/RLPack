//
// Created by Kartik Rajeshwaran on 2022-07-08.
//


#include "Dlqn1dOptions.h"

namespace model::dqn{

    Dlqn1dOptions::Dlqn1dOptions() = default;

    Dlqn1dOptions::Dlqn1dOptions(
            int32_t sequenceLength, std::vector<int32_t> &hiddenSizes,
            int32_t numAction, std::optional<std::shared_ptr<ActivationBase>> activation, float_t dropout
    ) : ModelOptionsBase(numAction, activation){

        sequenceLength_ = sequenceLength;
        hiddenSizes_ = hiddenSizes;
        numAction_ = numAction;
        activation_ = activation;
        dropout_ = dropout;

    }

    void Dlqn1dOptions::sequence_length(int32_t sequenceLength) {
        sequenceLength_ = sequenceLength;
    }

    void Dlqn1dOptions::hidden_sizes(std::vector<int32_t> &hiddenSizes) {
        hiddenSizes_ = hiddenSizes;
    }

    void Dlqn1dOptions::dropout(float_t dropout) {
        dropout_ = dropout;
    }

    int32_t Dlqn1dOptions::get_sequence_length() const {
        return sequenceLength_;
    }

    std::vector<int32_t> Dlqn1dOptions::get_hidden_sizes() {
        return hiddenSizes_;
    }

    float_t Dlqn1dOptions::get_dropout() const {
        return dropout_;
    }

    Dlqn1dOptions::~Dlqn1dOptions() = default;
}
