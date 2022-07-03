//
// Created by Kartik Rajeshwaran on 2022-07-01.
//

#ifndef RLPACK_DQN_DLQN1D_DLQN1D_H
#define RLPACK_DQN_DLQN1D_DLQN1D_H

#include "../DqnBase.hpp"

namespace dqn {

    template<class Activation = torch::nn::ReLU>
    class Dlqn1d : public DqnBase {
    public:
        Dlqn1d(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
             std::optional<Activation> activation = std::optional<Activation>(), float_t dropout = 0.5);

        ~Dlqn1d() override;

        torch::Tensor forward(torch::Tensor x) override;

    private:
        int32_t sequenceLength_;
        std::vector<int32_t> hiddenSizes_;
        float_t dropout_;
        uint32_t numBlocks_;

        torch::nn::ModuleList linearSubmodules = torch::nn::ModuleList();
        Activation activationSubmodule;
        std::shared_ptr<torch::nn::Flatten> flattenSubmodule = nullptr;
        std::shared_ptr<torch::nn::Dropout> dropoutSubmodule = nullptr;

        void setup_model();
    };

}


#endif //RLPACK_DQN_DLQN1D_DLQN1D_H
