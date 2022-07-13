//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_DLQN1DOPTIONS_H
#define RLPACK_DLQN1DOPTIONS_H

#include <torch/nn.h>
#include "../../../utils/Options/ModelOptions/ModelOptionsBase.h"

namespace model::dqn {

    class Dlqn1dOptions : public ModelOptionsBase {
    private:
        int32_t sequenceLength_{};
        std::vector<int32_t> hiddenSizes_;
        float_t dropout_ = 0.5;
    public:
        Dlqn1dOptions();

        Dlqn1dOptions(
                int32_t sequenceLength, std::vector<int32_t> &hiddenSizes,
                int32_t numAction,
                std::optional<std::shared_ptr<ActivationBase>> activation = std::optional<std::shared_ptr<ActivationBase>>(),
                float_t dropout = 0.5
        );

        ~Dlqn1dOptions();

        void sequence_length(int32_t sequenceLength);

        void hidden_sizes(std::vector<int32_t> &hiddenSizes);

        void dropout(float_t dropout);

        [[nodiscard]] int32_t get_sequence_length() const;

        std::vector<int32_t> get_hidden_sizes();

        [[nodiscard]] float_t get_dropout() const;

    };


}

#endif //RLPACK_DLQN1DOPTIONS_H
