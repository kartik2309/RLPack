//
// Created by Kartik Rajeshwaran on 2023-01-03.
//

#include "RolloutBufferData.h"
RolloutBufferData::RolloutBufferData() = default;

RolloutBufferData::~RolloutBufferData() = default;

std::map<std::string, torch::Tensor> RolloutBufferData::get_transition_data(torch::TensorOptions& tensorOptions) const {
    /*!
     * Gets the transition data of the current instance. Must have valid pointers set to state_current, rewards and dones. If
     * valid pointers are not set, an error is raised. This method applies tensor options to the tensors.
     *
     * @param tensorOptions : The tensor options to be applied to transition quantities.
     *
     * @return : The map of transitions outputs.
     */
    if ((stateCurrent == nullptr) or (stateNext == nullptr) or (reward == nullptr) or (done == nullptr)) {
        throw std::runtime_error("One of the quantities required for getting transition data was null!");
    }
    auto stateCurrent_ = stateCurrent->to(tensorOptions);
    auto stateNext_ = stateCurrent->to(tensorOptions);
    auto reward_ = reward->to(tensorOptions);
    auto done_ = done->to(tensorOptions);
    return {
            {STATE_CURRENT, stateCurrent_},
            {STATE_NEXT, stateNext_},
            {REWARD, reward_},
            {DONE, done_},
    };
}

std::map<std::string, torch::Tensor> RolloutBufferData::get_policy_output_data() const {
    /*!
     * Gets the policy output data of the current instance. Must have valid pointers set to action_log_probability,
     * state_current_value and entropy. If valid pointers are not set, an error is raised. This method does not
     * apply tensor options to the tensors.
     *
     * @return : The map of policy outputs.
     */
    if ((actionLogProbability == nullptr) or (stateCurrentValue == nullptr) or (stateNextValue == nullptr) or (entropy == nullptr)) {
        throw std::runtime_error("One of the quantities required for getting policy output data was null!");
    }
    return {
            {ACTION_LOG_PROBABILITY, *actionLogProbability},
            {STATE_CURRENT_VALUE, *stateCurrentValue},
            {STATE_NEXT_VALUE, *stateNextValue},
            {ENTROPY, *entropy}};
}
