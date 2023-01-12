//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#include "C_RolloutBuffer.h"

C_RolloutBuffer::C_RolloutBuffer(int64_t bufferSize,
                                 std::string& device,
                                 std::string& dtype,
                                 std::map<std::string, c10::intrusive_ptr<c10d::ProcessGroup>>& processGroupMap,
                                 const std::chrono::duration<int32_t>& workTimeoutDuration) {
    /*!
     * Class constructor for C_RolloutBuffer. This will allocate necessary memory as per input, and set relevant
     * attributes.
     *
     * @param bufferSize: The buffer size to be used.
     * @param device: The device on which PyTorch tensors are to be processed.
     * @param dtype: The datatype which is to be used for PyTorch tensors.
     * @param processGroupMap: The map of process group's intrusive pointer from Python to perform collective operations
     *  if required. If the RolloutBuffer is not being used in distributed setting, this can be passed as a map of
     *  nullptr like: {{"process_group", nullptr}}. This map must contain the ProcessGroup with the key `process_group`. A map
     *  is being used to take `TensorMap`  objected which is opaque and binded to avoid casting and easy pass by reference.
     * @param workTimeoutDuration: If a valid `processGroupMap` is being passed, this argument is relevant and indicates
     *  the work timeout duration in minutes. The work instrusive pointer explicitly calls wait to ensure synchronization.
     *  Default is set to 30 seconds.
     */
    processGroup_ = processGroupMap["process_group"];
    if (processGroup_.defined()) {
        bufferSize *= processGroup_->getSize();
    }
    rolloutBufferContainer_ = new RolloutBufferContainer(bufferSize, device, dtype);
    tensorOptions_ = rolloutBufferContainer_->get_tensor_options();
    workTimeoutDuration_ = std::chrono::duration_cast<std::chrono::milliseconds>(workTimeoutDuration);
}

/*!
 * Default destructor C_RolloutBuffer
 */
C_RolloutBuffer::~C_RolloutBuffer() {
    delete rolloutBufferContainer_;
}

void C_RolloutBuffer::insert_transition(std::map<std::string, torch::Tensor>& inputMap) {
    /*!
     * Insertion method to insert transitions into Rollout Buffer.
     *
     * @param inputMap: The input map of tensors with quantities to insert into the rollout buffer.
     * The map must contain the following keys:
     *  - state_current: The current state.
     *  - state_next: The next state.
     *  - done: The current done flag for episode termination/truncation.
     *  - reward: The reward obtained by the agent.
     *
     * For more information, please refer rlpack._C.rollout_buffer.RolloutBuffer and
     * rlpack.actor_critic.base.ActorCriticAgent
     */
    rolloutBufferContainer_->insert_transition(inputMap[STATE_CURRENT],
                                               inputMap[STATE_NEXT],
                                               inputMap[REWARD],
                                               inputMap[DONE]);
}

void C_RolloutBuffer::insert_policy_output(std::map<std::string, torch::Tensor>& inputMap) {
    /*!
     * Insertion method to insert policy outputs into Rollout Buffer.
     *
     * @param inputMap: The input map of tensors with quantities to insert_policy_output into the rollout buffer.
     * The map must contain the following keys:
     *  - action_log_probability: The log probability of the sampled action in the given distribution.
     *  - state_current_value: Current state value.
     *  - state_next_value: Value of next state.
     *  - entropy: Current entropy of the distribution.
     *
     * For more information, please refer rlpack._C.rollout_buffer.RolloutBuffer and
     * rlpack.actor_critic.base.ActorCriticAgent
     */
    rolloutBufferContainer_->insert_policy_output(inputMap[ACTION_LOG_PROBABILITY],
                                                  inputMap[STATE_CURRENT_VALUE],
                                                  inputMap[STATE_NEXT_VALUE],
                                                  inputMap[ENTROPY]);
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::compute_returns(float_t gamma) {
    /*!
     * Computes the returns with accumulated rewards.
     *
     * @param gamma: The discounting factor to be used for computing rewards.
     * @return A map of PyTorch tensor of returns, keyed "returns"
     */
    auto rewards = rolloutBufferContainer_->get_rewards_reference();
    auto totalRewards = rewards.size();
    std::vector<torch::Tensor> _returns(totalRewards);
    auto _r = torch::zeros({}, tensorOptions_);
    for (uint64_t index = totalRewards - 1; index != -1; index--) {
        _r = rewards[index] + (gamma * _r);
        _returns[index] = _r;
    }
    auto returns = torch::stack(_returns).to(tensorOptions_);
    return {{RETURNS, returns}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::compute_discounted_td_residuals(float_t gamma) {
    /*!
     *
     */
    auto rewards = rolloutBufferContainer_->get_rewards_reference();
    auto stateCurrentValues = rolloutBufferContainer_->get_state_current_values_reference();
    auto dones = rolloutBufferContainer_->get_dones_reference();
    auto totalResiduals = stateCurrentValues.size();
    auto _r = stateCurrentValues.back();
    std::vector<torch::Tensor> _tdResiduals(totalResiduals);
    for (uint64_t index = totalResiduals - 1; index != -1; index--) {
        _tdResiduals[index] = rewards[index] + (gamma * _r * (1 - dones[index])) - stateCurrentValues[index];
        _r = stateCurrentValues[index];
    }
    auto tdResiduals = torch::stack(_tdResiduals).to(tensorOptions_);
    return {{TD_RESIDUALS, tdResiduals}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::compute_generalized_advantage_estimates(float_t gamma,
                                                                                              float_t gaeLambda) {
    auto rewards = rolloutBufferContainer_->get_rewards_reference();
    auto stateCurrentValues = rolloutBufferContainer_->get_state_current_values_reference();
    auto dones = rolloutBufferContainer_->get_dones_reference();
    auto totalAdvantages = stateCurrentValues.size();
    std::vector<torch::Tensor> _advantages(totalAdvantages);
    auto _tdr = stateCurrentValues.back();
    auto _r = torch::zeros({}, tensorOptions_);
    auto delta = torch::zeros({}, tensorOptions_);
    for (uint64_t index = totalAdvantages - 1; index != -1; index--) {
        delta = rewards[index] + (gamma * _tdr * (1 - dones[index])) - stateCurrentValues[index];
        _r = delta + (gamma * gaeLambda * _r * (1 - dones[index]));
        _tdr = stateCurrentValues[index];
        _advantages[index] = _r;
    }
    auto advantages = torch::stack(_advantages).to(tensorOptions_);
    return {{ADVANTAGES, advantages}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_states_current() {
    return {{STATES_CURRENT,
             torch::stack(rolloutBufferContainer_->get_states_current_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_states_next() {
    return {{STATES_NEXT,
             torch::stack(rolloutBufferContainer_->get_states_next_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_states_statistics() {
    auto statesStacked = torch::concat({get_stacked_states_current()[STATES_CURRENT],
                                        get_stacked_states_next()[STATES_NEXT]});
    return {
            {"mean", statesStacked.mean(0)},
            {"std", statesStacked.std(0)},
            {"min", std::get<0>(statesStacked.min(0))},
            {"max", std::get<0>(statesStacked.max(0))}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_rewards() {
    /*!
     * Stacks the accumulated rewards and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of rewards, keyed "rewards".
     */
    return {{REWARDS,
             torch::stack(rolloutBufferContainer_->get_rewards_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_dones() {
    return {{DONES,
             torch::stack(rolloutBufferContainer_->get_dones_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_action_log_probabilities() {
    /*!
     * Stacks the accumulated action log probabilities and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of log of action probabilities, keyed "action_log_probabilities".
     */
    return {{ACTION_LOG_PROBABILITIES,
             torch::stack(rolloutBufferContainer_->get_action_log_probabilities_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_state_current_values() {
    /*!
     * Stacks the accumulated current state values and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of current states, keyed "state_current_values".
     */
    return {{STATE_CURRENT_VALUES,
             torch::stack(rolloutBufferContainer_->get_state_current_values_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_state_next_values() {
    /*!
     * Stacks the accumulated next state values and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of next states, keyed "state_next_values".
     */
    return {{STATE_NEXT_VALUES,
             torch::stack(rolloutBufferContainer_->get_state_next_values_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_entropies() {
    /*!
     * Stacks the accumulated current entropies and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of entropies, keyed "entropies".
     */
    return {{ENTROPIES,
             torch::stack(rolloutBufferContainer_->get_entropies_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::transition_at(int64_t index) {
    /*!
     * Obtain the transitions at a given index.
     *
     * @param index : The index of the transitions buffer.
     * @return : The map with transitions quantities viz:
     *  - state_current
     *  - state_next
     *  - reward
     *  - done
     */
    auto data = rolloutBufferContainer_->transition_at(index);
    return data.get_transition_data(tensorOptions_);
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::policy_output_at(int64_t index) {
    /*!
     * Obtain the policy output at a given index.
     *
     * @param index : The index of the policy output buffer.
     * @return : The map with policy output quantities viz:
     *  - action_log_probability
     *  - state_current_value
     *  - state_next_value
     *  - entropy
     */
    auto data = rolloutBufferContainer_->policy_output_at(index);
    return data.get_policy_output_data();
}

void C_RolloutBuffer::clear_transitions() {
    /*!
     * Clears the transition vectors.
     */
    rolloutBufferContainer_->clear_transitions();
}

void C_RolloutBuffer::clear_policy_outputs() {
    /*!
     * Clears the policy output vectors
     */
    rolloutBufferContainer_->clear_policy_outputs();
}

size_t C_RolloutBuffer::size_transitions() {
    /*!
     * Returns the size of Rollout Buffer.
     */
    return rolloutBufferContainer_->size_transitions();
}

size_t C_RolloutBuffer::size_policy_outputs() {
    /*!
     * Returns the size of Rollout Buffer.
     */
    return rolloutBufferContainer_->size_policy_outputs();
}

void C_RolloutBuffer::extend_transitions() {
    /*!
     * This method extends the transitions by performing gather for transition quantities. This method will throw a
     * runtime error if process group is not defined.
     */
    if (not processGroup_.defined()) {
        throw std::runtime_error("Encountered nullptr for process group!");
    }
    auto statesCurrentGathered = gather_with_process_group_(get_stacked_states_current()[STATES_CURRENT]);
    auto statesNextGathered = gather_with_process_group_(get_stacked_states_next()[STATES_NEXT]);
    auto rewardsGathered = gather_with_process_group_(get_stacked_rewards()[REWARDS]);
    auto donesGathered = gather_with_process_group_(get_stacked_dones()[DONES]);
    // Directly create the extension map.
    std::map<std::string, std::vector<torch::Tensor>> extensionMap = {{STATES_CURRENT, statesCurrentGathered},
                                                                      {STATES_NEXT, statesNextGathered},
                                                                      {REWARDS, rewardsGathered},
                                                                      {DONES, donesGathered}};
    if (extensionMap[STATES_CURRENT].empty()) {
        return;
    }
    rolloutBufferContainer_->extend_transitions(extensionMap);
}

void C_RolloutBuffer::set_transitions_iterator(int64_t batchSize) {
    auto dataset = RolloutBufferBatch(rolloutBufferContainer_, tensorOptions_);
    auto mapDataset = dataset.map(RolloutBufferBatchTransform(batchSize));
    dataloader_ = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(mapDataset,
                                                                                          batchSize);
}

C_RolloutBuffer::DataLoader& C_RolloutBuffer::get_dataloader_reference() {
    return dataloader_;
}

std::vector<torch::Tensor> C_RolloutBuffer::gather_with_process_group_(torch::Tensor& inputTensor) {
    /*!
     * Performs gather with the initialized process group. This method will throw a runtime error
     * if process group is not defined.
     *
     * @param inputTensor : The input tensor which is to be used for input in gather. This tensor is broadcasted to
     *  other processes in the group and receives the same tensor instances from other groups.
     * @return : A vector of tensors is returned from groups. The vector will have tensors from other processes only and will
     * be unbinded.
     */
    if (not processGroup_.defined()) {
        throw std::runtime_error("Encountered nullptr for process group!");
    }
    // Create result vector.
    std::vector<torch::Tensor> gatherUnbindedTensors;
    // Get current world size and rank.
    auto worldSize = processGroup_->getSize();
    auto rank = processGroup_->getRank();
    // Prepare input tensors for gather operation.
    std::vector<at::Tensor> inputTensors = {inputTensor};
    // Define output tensor for gather operation.
    std::vector<std::vector<at::Tensor>> outputTensors;
    // Reserve memory in vector for master process.
    gatherUnbindedTensors.reserve(rolloutBufferContainer_->get_buffer_size());
    // Initialize output vector with size 1.
    outputTensors = std::vector<std::vector<at::Tensor>>(1);
    // Prepare placeholder tensor for outputs in gather in master process.
    auto placeHolderZeroTensor = torch::zeros(inputTensor.sizes(), tensorOptions_);
    // Initialize output vector with world size at zeroth index.
    outputTensors[0] = std::vector<torch::Tensor>(worldSize);
    // Place placeholder tensors in output vector.
    for (uint64_t index = 0; index < worldSize; index++) {
        outputTensors[0][index] = placeHolderZeroTensor;
    }
    // Call gather in process group.
    auto workPointer = processGroup_->allgather(outputTensors, inputTensors);
    // Use wait clause to force synchronization and wait for all processes to finish or raise an error.
    workPointer->wait(workTimeoutDuration_);
    // Fill the result vector with tensors for master process.
    for (uint64_t index = 0; index < worldSize; index++) {
        if (index != rank) {
            auto unbindedVector = torch::unbind(outputTensors[0][index]);
            for (uint64_t unbiddenIndex = 0; unbiddenIndex < unbindedVector.size(); unbiddenIndex++) {
                gatherUnbindedTensors.push_back(unbindedVector[unbiddenIndex]);
            }
        }
    }
    return gatherUnbindedTensors;
}
