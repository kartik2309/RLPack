//
// Created by Kartik Rajeshwaran on 2022-06-20.
//

#ifndef RLPACK_SRC_AGENTIMPL_HPP_
#define RLPACK_SRC_AGENTIMPL_HPP_

#include <torch/torch.h>
#include <boost/log/trivial.hpp>

namespace agent {
    class AgentBase {

    protected:
        std::map<torch::ScalarType, int> dTypeCodes = {
                {torch::kFloat64, 0},
                {torch::kDouble, 1},
                {torch::kFloat32, 2},
        };

    public:

        AgentBase();

        virtual int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done);

        virtual int policy(torch::Tensor &stateCurrent);

        virtual void save();

        virtual void load();

        virtual void finish();

        virtual void barrier();

        virtual void sync_models();
    };
}

#endif//RLPACK_SRC_AGENTIMPL_HPP_
