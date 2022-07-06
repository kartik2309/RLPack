//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_SRC_DQN_DQNBASE_HPP_
#define RLPACK_SRC_DQN_DQNBASE_HPP_

#include <torch/torch.h>

namespace dqn {
    class DqnBase : public torch::nn::Module {

    public:
        virtual torch::Tensor forward(torch::Tensor x);

        ~DqnBase() override;
    };

    torch::Tensor DqnBase::forward(torch::Tensor x) {
        return x;
    }

    DqnBase::~DqnBase() = default;
}// namespace dqn

#endif//RLPACK_SRC_DQN_DQNBASE_HPP_
