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
    // Initialize private variables.
    bufferSize_ = bufferSize;
    device_ = UtilityMapping::deviceMap[device];
    dtype_ = UtilityMapping::dTypeMap[dtype];
    tensorOptions_ = torch::TensorOptions().device(device_).dtype(dtype_);
    // Initialize all vectors.
    rewards_ = std::vector<torch::Tensor>(bufferSize_);
    statesCurrent_ = std::vector<torch::Tensor>(bufferSize_);
    statesNext_ = std::vector<torch::Tensor>(bufferSize_);
    dones_ = std::vector<torch::Tensor>(bufferSize_);
    actionLogProbabilities_ = std::vector<torch::Tensor>(bufferSize_);
    stateCurrentValues_ = std::vector<torch::Tensor>(bufferSize_);
    stateNextValues_ = std::vector<torch::Tensor>(bufferSize_);
    entropies_ = std::vector<torch::Tensor>(bufferSize_);
    transitionData_ = std::vector<RolloutBufferData>(bufferSize_);
    policyOutputData_ = std::vector<RolloutBufferData>(bufferSize_);
    // Initialize all counters.
    transitionCounter_ = 0;
    policyOutputCounter_ = 0;
    // Reserve memory for reference vector.
    reference_.reserve(bufferSize);
}

/*!
 * Default destructor for RolloutBufferContainer
 */
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
    statesCurrent_[transitionCounter_] = stateCurrent;
    statesNext_[transitionCounter_] = stateNext;
    rewards_[transitionCounter_] = reward;
    dones_[transitionCounter_] = done;
    transitionData.stateCurrent = &statesCurrent_[transitionCounter_];
    transitionData.stateNext = &statesNext_[transitionCounter_];
    transitionData.reward = &rewards_[transitionCounter_];
    transitionData.done = &dones_[transitionCounter_];
    transitionData_[transitionCounter_] = transitionData;
    transitionCounter_ += 1;
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
    actionLogProbabilities_[policyOutputCounter_] = actionLogProbability;
    stateCurrentValues_[policyOutputCounter_] = stateCurrentValue;
    stateNextValues_[policyOutputCounter_] = stateNextValue;
    entropies_[policyOutputCounter_] = entropy;
    policyOutputData.actionLogProbability = &actionLogProbabilities_[policyOutputCounter_];
    policyOutputData.stateCurrentValue = &stateCurrentValues_[policyOutputCounter_];
    policyOutputData.stateCurrentValue = &stateNextValues_[policyOutputCounter_];
    policyOutputData.entropy = &entropies_[policyOutputCounter_];
    policyOutputData_[policyOutputCounter_] = policyOutputData;
    policyOutputCounter_ += 1;
}

void RolloutBufferContainer::clear_transitions() {
    /*!
     * Clears the transition vectors. Doesn't clear the vector but changes the index causing insertion
     * operations to overwrite the previous values.
     */
    transitionCounter_ = 0;
}

void RolloutBufferContainer::clear_policy_outputs() {
    /*!
     * Clears the policy output vectors. Doesn't clear the vector but changes the index causing insertion
     * operations to overwrite the previous values.
     */
    policyOutputCounter_ = 0;
}

RolloutBufferData RolloutBufferContainer::transition_at(int64_t index) {
    /*!
     * Obtain the transitions at a given index.
     *
     * @param index : The index of the transitions buffer.
     * @return An instance of RolloutBufferData with correct instances set.
     */
    if (index > transitionCounter_) {
        throw std::out_of_range("Given index is larger than the current size of RolloutBufferContainer!");
    }
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
    if (index > policyOutputCounter_) {
        throw std::out_of_range("Given index is larger than the current size of RolloutBufferContainer!");
    }
    auto data = policyOutputData_[index];
    return data;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_states_current_reference() {
    /*!
     * Gets the reference of `states_current` buffer.
     * 
     * @return The reference to `states_current`
     */
    fill_reference_vector_(statesCurrent_, size_transitions());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_states_next_reference() {
    /*!
     * Gets the reference of `states_next` buffer.
     *
     * @return The reference to `states_next`
     */
    fill_reference_vector_(statesNext_, size_transitions());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_rewards_reference() {
    /*!
     * Gets the reference of `rewards` buffer.
     * 
     * @return The reference to `rewards`
     */
    fill_reference_vector_(rewards_, size_transitions());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_dones_reference() {
    /*!
     * Gets the reference of `dones` buffer.
     * 
     * @return The reference to `dones`
     */
    fill_reference_vector_(dones_, size_transitions());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_action_log_probabilities_reference() {
    /*!
     * Gets the reference of `action_log_probabilities` buffer.
     * 
     * @return The reference to `action_log_probabilities`
     */
    fill_reference_vector_(actionLogProbabilities_, size_policy_outputs());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_state_current_values_reference() {
    /*!
     * Gets the reference of `state_current_values` buffer.
     * 
     * @return The reference to `state_current_values`
     */
    fill_reference_vector_(stateCurrentValues_, size_policy_outputs());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_state_next_values_reference() {
    /*!
     * Gets the reference of `state_next_values` buffer.
     *
     * @return The reference to `state_next_values`
     */
    fill_reference_vector_(stateNextValues_, size_policy_outputs());
    return reference_;
}

std::vector<torch::Tensor> &RolloutBufferContainer::get_entropies_reference() {
    /*!
     * Gets the reference of `entropies` buffer.
     * 
     * @return The reference to `entropies`
     */
    fill_reference_vector_(entropies_, size_policy_outputs());
    return reference_;
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

uint64_t RolloutBufferContainer::size_transitions() const {
    /*!
     * The size of transitions buffer.
     *
     * @return The transitions buffer size.
     */
    return transitionCounter_;
}

uint64_t RolloutBufferContainer::size_policy_outputs() const {
    /*!
     * The size of policy outputs buffer.
     *
     * @return The policy outputs' buffer size.
     */
    return policyOutputCounter_;
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
    auto additionalSize = extensionMap[DONES].size();
    auto enableParallelism = OMP_PARALLELISM_THRESHOLD < additionalSize;
    {
        // Parallel loop to extend transitions.
#pragma omp parallel for if (enableParallelism) default(none) \
        firstprivate(additionalSize)                                  \
                shared(extensionMap, transitionCounter_, statesCurrent_, statesNext_, rewards_, dones_, transitionData_)
        for (uint64_t index = 0; index < additionalSize; index++) {
            auto transitionData = RolloutBufferData();
            statesCurrent_[transitionCounter_ + index] = extensionMap[STATES_CURRENT][index];
            statesNext_[transitionCounter_ + index] = extensionMap[STATES_NEXT][index];
            rewards_[transitionCounter_ + index] = extensionMap[REWARDS][index];
            dones_[transitionCounter_ + index] = extensionMap[DONES][index];
            transitionData.stateCurrent = &statesCurrent_[index];
            transitionData.stateNext = &statesNext_[index];
            transitionData.reward = &rewards_[index];
            transitionData.done = &dones_[index];
            transitionData_[transitionCounter_ + index] = transitionData;
        }
    }
    transitionCounter_ += additionalSize;
}

void RolloutBufferContainer::fill_reference_vector_(std::vector<torch::Tensor> &source, uint64_t size) {
    /*!
     * Private method to fill the reference container to obtain the requested references of correct size.
     *
     * @param source: The reference to input source vector.
     * @param size: The current size of buffer (transition or policy output).
     */
    reference_.clear();
    for (uint64_t index = 0; index != size; index++) {
        reference_.push_back(source[index]);
    }
}
