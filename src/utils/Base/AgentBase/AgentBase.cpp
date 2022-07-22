//
// Created by Kartik Rajeshwaran on 2022-07-12.
//

#include "AgentBase.h"

namespace agent {

    int AgentBase::train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done) {
        BOOST_LOG_TRIVIAL(warning) << "AgentBase class cannot train model. "
                                      "This method must be overridden. "
                                      "Will return 0"
                                   << std::endl;
        return 0;
    }

    int AgentBase::policy(torch::Tensor &stateCurrent) {
        BOOST_LOG_TRIVIAL(warning) << "AgentBase class cannot policy model. "
                                      "This method must be overridden. "
                                      "Will return 0"
                                   << std::endl;
        return 0;
    }

    void AgentBase::save() {
        throw std::runtime_error("AgentBase class cannot save model. This method must be overridden");
    }

    void AgentBase::load() {
        throw std::runtime_error("AgentBase class cannot load model. This method must be overridden");
    }

    void AgentBase::finish() {
        throw std::runtime_error("AgentBase class cannot finish MPI Comm. This method must be overridden");
    }

    void AgentBase::barrier() {
        throw std::runtime_error("AgentBase class cannot provide MPI Barrier. This method must be overridden");
    }

    void AgentBase::sync_models() {
        throw std::runtime_error("AgentBase class cannot provide MPI Barrier. This method must be overridden");
    }

    AgentBase::AgentBase() = default;
}