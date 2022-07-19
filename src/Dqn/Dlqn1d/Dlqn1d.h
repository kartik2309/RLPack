//
// Created by Kartik Rajeshwaran on 2022-07-01.
//

#ifndef RLPACK_DQN_DLQN1D_DLQN1D_H
#define RLPACK_DQN_DLQN1D_DLQN1D_H

#include <torch/torch.h>
#include "Dlqn1dOptions/Dlqn1dOptions.h"
#include "../../utils/Base/ModelBase/ModelBase.h"
#include "../../utils/Activations/Activations.hpp"

namespace model::dqn {

    class Dlqn1d : public model::ModelBase {
    public:
        Dlqn1d(
                int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                std::optional<std::shared_ptr<ActivationBase>> activation = std::optional<std::shared_ptr<ActivationBase>>(),
                float_t dropout = 0.5
        );

        explicit Dlqn1d(std::unique_ptr<Dlqn1dOptions> &options);

        ~Dlqn1d() override;

        torch::Tensor forward(torch::Tensor x) override;

    private:
        int32_t sequenceLength_;
        std::vector<int32_t> hiddenSizes_;
        float_t dropout_;
        uint32_t numBlocks_;

        torch::nn::ModuleList linearSubmodules = torch::nn::ModuleList();
        std::shared_ptr<ActivationBase> activationSubmodule;;
        std::shared_ptr<torch::nn::Flatten> flattenSubmodule = nullptr;
        std::shared_ptr<torch::nn::Dropout> dropoutSubmodule = nullptr;

        void setup_model();
    };

}//namespace model::dqn

#endif //RLPACK_DQN_DLQN1D_DLQN1D_H
