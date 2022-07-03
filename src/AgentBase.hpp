//
// Created by Kartik Rajeshwaran on 2022-06-20.
//

#ifndef RLPACK_SRC_AGENTIMPL_HPP_
#define RLPACK_SRC_AGENTIMPL_HPP_

#include <torch/torch.h>
#include <boost/log/trivial.hpp>

class AgentBase {

public:

    AgentBase();

    virtual int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done);

    virtual int policy(torch::Tensor &stateCurrent);

    virtual void save();
};

int AgentBase::train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done) {
    BOOST_LOG_TRIVIAL(warning) << "AgentBase class cannot train models. "
                                  "This method must be overridden. "
                                  "Will return 0"
                               << std::endl;
    return 0;
}

int AgentBase::policy(torch::Tensor &stateCurrent) {
    BOOST_LOG_TRIVIAL(warning) << "AgentBase class cannot policy models. "
                                  "This method must be overridden. "
                                  "Will return 0"
                               << std::endl;
    return 0;
}

void AgentBase::save() {
    throw std::runtime_error("AgentBase class cannot save models. This method must be overridden");
}

AgentBase::AgentBase() = default;

#endif//RLPACK_SRC_AGENTIMPL_HPP_
