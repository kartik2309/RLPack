//
// Created by Kartik Rajeshwaran on 2023-01-03.
//

#include "RolloutBufferContainer.h"

RolloutBufferContainer::RolloutBufferContainer(int64_t bufferSize, std::string &device, std::string &dtype) {
    /*!
     * Class constructor for RolloutBufferContainer. This will allocate necessary memory as per input, and set relevant
     * attributes.
     *
     * @param bufferSize: The buffer size to be used.
     * @param device: The device on which PyTorch tensors are to be processed.
     * @param dtype: The datatype which is to be used for PyTorch tensors.
     */
    bufferSize_ = bufferSize;
    device_ = Maps::deviceMap[device];
    dtype_ = Maps::dTypeMap[dtype];
    tensorOptions_ = torch::TensorOptions().device(device_).dtype(dtype_);
    rewards_.reserve(bufferSize_);
    statesCurrent_.reserve(bufferSize_);
    statesNext_.reserve(bufferSize_);
    dones_.reserve(bufferSize_);
    actionLogProbabilities_.reserve(bufferSize_);
    stateCurrentValues_.reserve(bufferSize_);
    stateNextValues_.reserve(bufferSize_);
    entropies_.reserve(bufferSize_);
    transitionData_.reserve(bufferSize_);
    policyOutputData_.reserve(bufferSize);
}

RolloutBufferContainer::~RolloutBufferContainer() = default;

void RolloutBufferContainer::insert_transition(torch::Tensor &stateCurrent,
                                               torch::Tensor &stateNext,
                                               torch::Tensor &reward,
                                               torch::Tensor &done) {
    /*!
     * Insertion method to insert transitions into Rollout Buffer Container.
     *
     * @param stateCurrent: The current state.
     * @param stateNext: The next state.
     * @param reward: The reward obtained by the agent.
     * @param done: The current done flag for episode termination/truncation.
     *
     * For more information, please refer rlpack._C.rollout_buffer.RolloutBuffer and
     * rlpack.actor_critic.base.ActorCriticAgent
     */
    auto transitionData = RolloutBufferData();
    statesCurrent_.push_back(stateCurrent);
    statesNext_.push_back(stateNext);
    rewards_.push_back(reward);
    dones_.push_back(done);
    transitionData.stateCurrent = &statesCurrent_.back();
    transitionData.stateNext = &statesNext_.back();
    transitionData.reward = &rewards_.back();
    transitionData.done = &dones_.back();
    transitionData_.push_back(transitionData);
}

void RolloutBufferContainer::insert_policy_output(torch::Tensor &actionLogProbability,
                                                  torch::Tensor &stateCurrentValue,
                                                  torch::Tensor &stateNextValue,
                                                  torch::Tensor &entropy) {
    /*!
     * Insertion method to insert policy outputs into Rollout Buffer Container.
     *
     * @param actionLogProbability: The log probability of the sampled action in the given distribution.
     * @param stateCurrentValue: Current state value.
     * @param stateNextValue: The next state value.
     * @param entropy: Current entropy of the distribution.
     *
     * For more information, please refer rlpack._C.rollout_buffer.RolloutBuffer and
     * rlpack.actor_critic.base.ActorCriticAgent
     */
    auto policyOutputData = RolloutBufferData();
    actionLogProbabilities_.push_back(actionLogProbability);
    stateCurrentValues_.push_back(stateCurrentValue);
    stateNextValues_.push_back(stateNextValue);
    entropies_.push_back(entropy);
    policyOutputData.actionLogProbability = &actionLogProbabilities_.back();
    policyOutputData.stateCurrentValue = &stateCurrentValues_.back();
    policyOutputData.stateCurrentValue = &stateNextValues_.back();
    policyOutputData.entropy = &entropies_.back();
    policyOutputData_.push_back(policyOutputData);
}

void RolloutBufferContainer::clear_transitions() {
    /*!
     * Clears the transition vectors.
     */
    rewards_.clear();
    statesCurrent_.clear();
    statesNext_.clear();
    dones_.clear();
    transitionData_.clear();
}

void RolloutBufferContainer::clear_policy_outputs() {
    /*!
     * Clears the policy output vectors
     */
    actionLogProbabilities_.clear();
    stateCurrentValues_.clear();
    stateNextValues_.clear();
    entropies_.clear();
    policyOutputData_.clear();
}

RolloutBufferData RolloutBufferContainer::transition_at(int64_t index) {
    /*!
     * Obtain the transitions at a given index.
     *
     * @param index : The index of the transitions buffer.
     * @return An instance of RolloutBufferData with correct instances set.
     */
    auto data = transitionData_[index];
    return data;
}

RolloutBufferData RolloutBufferContainer::policy_output_at(int64_t index) {
    /*!
     * Obtain the policy output at a given index.
     *
     * @param index : The index of the policy output buffer.
     * @return An instance of RolloutBufferData with correct instances set.
     */
    auto data = policyOutputData_[index];
    return data;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_states_current_reference() {
    /*!
     * Gets the reference of `states_current` buffer.
     * 
     * @return The reference to `states_current`
     */
    return statesCurrent_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_states_next_reference() {
    /*!
     * Gets the reference of `states_next` buffer.
     *
     * @return The reference to `states_next`
     */
    return statesNext_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_rewards_reference() {
    /*!
     * Gets the reference of `rewards` buffer.
     * 
     * @return The reference to `rewards`
     */
    return rewards_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_dones_reference() {
    /*!
     * Gets the reference of `dones` buffer.
     * 
     * @return The reference to `dones`
     */
    return dones_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_action_log_probabilities_reference() {
    /*!
     * Gets the reference of `action_log_probabilities` buffer.
     * 
     * @return The reference to `action_log_probabilities`
     */
    return actionLogProbabilities_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_state_current_values_reference() {
    /*!
     * Gets the reference of `state_current_values` buffer.
     * 
     * @return The reference to `state_current_values`
     */
    return stateCurrentValues_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_state_next_values_reference() {
    /*!
     * Gets the reference of `state_next_values` buffer.
     *
     * @return The reference to `state_next_values`
     */
    return stateNextValues_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_entropies_reference() {
    /*!
     * Gets the reference of `entropies` buffer.
     * 
     * @return The reference to `entropies`
     */
    return entropies_;
}
torch::TensorOptions RolloutBufferContainer::get_tensor_options() {
    /*!
     * Gets the tensor options being used.
     * 
     * @return The TensorOptions object.
     */
    return tensorOptions_;
}

int64_t RolloutBufferContainer::get_buffer_size() const {
    /*!
     * Gets the current buffer size. This also indicates the capacity.
     *
     * @return The buffer size.
     */
    return bufferSize_;
}

size_t RolloutBufferContainer::size_transitions() {
    /*!
     * The size of transitions buffer.
     *
     * @return The transitions buffer size.
     */
    return transitionData_.size();
}

size_t RolloutBufferContainer::size_policy_outputs() {
    /*!
     * The size of policy outputs buffer.
     *
     * @return The policy outputs' buffer size.
     */
    return policyOutputData_.size();
}

void RolloutBufferContainer::extend_transitions(std::map<std::string, std::vector<torch::Tensor>> &extensionMap) {
    /*!
     * Extends the transitions as per given extension map.
     *
     * @param extensionMap: The map of transition quantities with corresponding vectors to be used to extend the transitions.
     *  The extensionMap must contain the mandatory keys viz:
     *      - states_current
     *      - states_next
     *      - rewards
     *      - dones
     */
    for (uint64_t index = 0; index < extensionMap[DONES].size(); index++) {
        insert_transition(extensionMap[STATES_CURRENT][index],
                          extensionMap[STATES_NEXT][index],
                          extensionMap[REWARDS][index],
                          extensionMap[DONES][index]);
    }
}
