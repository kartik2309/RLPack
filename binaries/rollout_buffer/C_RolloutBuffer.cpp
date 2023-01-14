//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#include "C_RolloutBuffer.h"

C_RolloutBuffer::C_RolloutBuffer(int64_t bufferSize,
                                 std::string& device,
                                 std::string& dtype) {
    /*!
     * Class constructor for C_RolloutBuffer. This will allocate necessary memory as per input, and set relevant
     * attributes. This method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.__init__.
     *
     *
     * @param bufferSize: The buffer size to be used.
     * @param device: The device on which PyTorch tensors are to be processed.
     * @param dtype: The datatype which is to be used for PyTorch tensors.
     */
    rolloutBufferContainer_ = new RolloutBufferContainer(bufferSize, device, dtype);
    tensorOptions_ = rolloutBufferContainer_->get_tensor_options();
}


C_RolloutBuffer::~C_RolloutBuffer() {
    /*!
    * Default destructor C_RolloutBuffer. This deletes the rolloutBufferContainer_
    * and deallocates the memory.
    */
    delete rolloutBufferContainer_;
}

void C_RolloutBuffer::insert_transition(std::map<std::string, torch::Tensor>& inputMap) {
    /*!
     * Insertion method to insert transitions into Rollout Buffer. This method is C++ backend
     * for rlpack._C.rollout_buffer.RolloutBuffer.insert_transition.
     *
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
     * Insertion method to insert policy outputs into Rollout Buffer. This method is C++ backend
     * for rlpack._C.rollout_buffer.RolloutBuffer.insert_policy_output.
     *
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
     * Computes the returns with accumulated rewards. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.compute_returns.
     *
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
     * Computes the discounted TD residuals for the given gamma. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.compute_discounted_td_residuals
     *
     *
     * @param gamma: The discounting factor value.
     * @return The tensor map with key "td_residuals" and corresponding TD residual values.
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
    /*!
     * Computes the Generalized Advantage Estimates; a bias-variance tradeoff. This is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.compute_generalized_advantage_estimates
     *
     *
     * @param gamma: The discounting factor value.
     * @param gaeLambda: The GAE Lambda value which controls the bias-variance tradeoff. When gaeLambda is 1,
     *  classic advantage is returned.
     * @return A tensor map with key "advantages" and corresponding tensor of advantages.
     */
    auto rewards = rolloutBufferContainer_->get_rewards_reference();
    auto stateCurrentValues = rolloutBufferContainer_->get_state_current_values_reference();
    auto dones = rolloutBufferContainer_->get_dones_reference();
    auto totalAdvantages = stateCurrentValues.size();
    std::vector<torch::Tensor> _advantages(totalAdvantages);
    auto _tdr = stateCurrentValues[totalAdvantages - 1];
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
    /*!
     * Stacks the accumulated states current and moves them to correct tensor options (for device and datatype). This
     * method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_states_current.
     *
     *
     * @return A map of PyTorch tensor of states current, keyed "states_current".
     */
    return {{STATES_CURRENT,
             torch::stack(rolloutBufferContainer_->get_states_current_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_states_next() {
    /*!
     * Stacks the accumulated states current and moves them to correct tensor options (for device and datatype). This
     * method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_states_next.
     *
     *
     * @return A map of PyTorch tensor of states next, keyed "states_next".
     */
    return {{STATES_NEXT,
             torch::stack(rolloutBufferContainer_->get_states_next_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_rewards() {
    /*!
     * Stacks the accumulated rewards and moves them to correct tensor options (for device and datatype). This
     * method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_rewards.
     *
     *
     * @return A map of PyTorch tensor of rewards, keyed "rewards".
     */
    return {{REWARDS,
             torch::stack(rolloutBufferContainer_->get_rewards_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_dones() {
    /*!
     * Stacks the accumulated dones and moves them to correct tensor options (for device and datatype). This
     * method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_dones.
     *
     *
     * @return A map of PyTorch tensor of dones, keyed "dones".
     */
    return {{DONES,
             torch::stack(rolloutBufferContainer_->get_dones_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_action_log_probabilities() {
    /*!
     * Stacks the accumulated action log probabilities and moves them to correct tensor options
     * (for device and datatype). This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_action_log_probabilities.
     *
     *
     * @return A map of PyTorch tensor of log of action probabilities, keyed "action_log_probabilities".
     */
    return {{ACTION_LOG_PROBABILITIES,
             torch::stack(rolloutBufferContainer_->get_action_log_probabilities_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_state_current_values() {
    /*!
     * Stacks the accumulated current state values and moves them to correct tensor options (for device and datatype).
     * This method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_state_current_values.
     *
     *
     * @return A map of PyTorch tensor of current states, keyed "state_current_values".
     */
    return {{STATE_CURRENT_VALUES,
             torch::stack(rolloutBufferContainer_->get_state_current_values_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_state_next_values() {
    /*!
     * Stacks the accumulated next state values and moves them to correct tensor options (for device and datatype).
     * This method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_state_next_values.
     *
     *
     * @return A map of PyTorch tensor of next states, keyed "state_next_values".
     */
    return {{STATE_NEXT_VALUES,
             torch::stack(rolloutBufferContainer_->get_state_next_values_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_entropies() {
    /*!
     * Stacks the accumulated current entropies and moves them to correct tensor options (for device and datatype).
     * This method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_stacked_entropies.
     *
     *
     * @return A map of PyTorch tensor of entropies, keyed "entropies".
     */
    return {{ENTROPIES,
             torch::stack(rolloutBufferContainer_->get_entropies_reference()).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_states_statistics() {
    /*!
     * Computes the statistics for accumulated states. Both states current and states next are used for
     * computing the statistics. The statistics are computed along each state dimension. This method is C++
     * backend for rlpack._C.rollout_buffer.RolloutBuffer.get_states_statistics.
     *
     *
     * @return A map of PyTorch tensors of statistics. The following keys are presents:
     *  - min: The minimum value across each state dimension.
     *  - max: The maximum value across each state dimension.
     *  - mean: The mean value across each state dimension.
     *  - std: The std value across each state dimension.
     */
    auto statesStacked = torch::concat({get_stacked_states_current()[STATES_CURRENT],
                                        get_stacked_states_next()[STATES_NEXT]});
    statesStacked = statesStacked.view({-1, statesStacked.size(-1)});
    return {{"mean", statesStacked.mean(0)},
            {"std", statesStacked.std(0)},
            {"min", std::get<0>(statesStacked.min(0))},
            {"max", std::get<0>(statesStacked.max(0))}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_advantage_statistics(float_t gamma, float_t gae_lambda) {
    /*!
     * Computes the statistics for computed advantage. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.get_advantage_statistics.
     *
     *
     * @return A map of PyTorch tensors of statistics. The following keys are presents:
     *  - min: The minimum value.
     *  - max: The maximum value.
     *  - mean: The mean value.
     *  - std: The std value.
     */
    auto advantages = compute_generalized_advantage_estimates(gamma, gae_lambda)[ADVANTAGES];
    return {{"mean", advantages.mean(0)},
            {"std", advantages.std(0)},
            {"min", std::get<0>(advantages.min(0))},
            {"max", std::get<0>(advantages.max(0))}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_action_log_probabilities_statistics() {
    /*!
     * Computes the statistics for accumulated action log probabilities. Statistics are computed across each
     * action dimension. This method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_action_log_probabilities_statistics.
     *
     *
     * @return A map of PyTorch tensors of statistics. The following keys are presents:
     *  - min: The minimum value across each action dimension.
     *  - max: The maximum value across each action dimension.
     *  - mean: The mean value across each action dimension.
     *  - std: The std value across each action dimension.
     */
    auto actionLogProbabilities = get_stacked_action_log_probabilities()[ACTION_LOG_PROBABILITIES];
    actionLogProbabilities = actionLogProbabilities.view({-1, actionLogProbabilities.size(-1)});
    return {{"mean", actionLogProbabilities.mean(0)},
            {"std", actionLogProbabilities.std(0)},
            {"min", std::get<0>(actionLogProbabilities.min(0))},
            {"max", std::get<0>(actionLogProbabilities.max(0))}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_state_values_statistics() {
    /*!
     * Computes the statistics for accumulated state values. Both state current values and state next values are
     * used for statistics computation. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.get_state_values_statistics.
     *
     *
     * @return A map of PyTorch tensors of statistics. The following keys are presents:
     *  - min: The minimum value.
     *  - max: The maximum value.
     *  - mean: The mean value.
     *  - std: The std value.
     */
    auto statesValuesStacked = torch::concat({get_stacked_state_current_values()[STATE_CURRENT_VALUES],
                                              get_stacked_state_next_values()[STATE_NEXT_VALUES]});
    statesValuesStacked = statesValuesStacked.view({-1, statesValuesStacked.size(-1)});
    return {{"mean", statesValuesStacked.mean(0)},
            {"std", statesValuesStacked.std(0)},
            {"min", std::get<0>(statesValuesStacked.min(0))},
            {"max", std::get<0>(statesValuesStacked.max(0))}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_entropy_statistics() {
    /*!
     * Computes the statistics for entropies. Statistics are computed across each action dimension. This method is
     * C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.get_entropy_statistics.
     *
     *
     * @return A map of PyTorch tensors of statistics. The following keys are presents:
     *  - min: The minimum value across each action dimension.
     *  - max: The maximum value across each action dimension.
     *  - mean: The mean value across each action dimension.
     *  - std: The std value across each action dimension.
     */
    auto entropies = get_stacked_entropies()[ENTROPIES];
    entropies = entropies.view({-1, entropies.size(-1)});
    return {{"mean", entropies.mean(0)},
            {"std", entropies.std(0)},
            {"min", std::get<0>(entropies.min(0))},
            {"max", std::get<0>(entropies.max(0))}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::transition_at(int64_t index) {
    /*!
     * Obtain the transitions at a given index. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.transition_at.
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
     * Obtain the policy output at a given index. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.policy_output_at.
     *
     * @param index : The index of the policy output buffer.
     * @return The map with policy output quantities viz:
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
     * Clears the transition vectors. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.clear_transitions.
     */
    rolloutBufferContainer_->clear_transitions();
}

void C_RolloutBuffer::clear_policy_outputs() {
    /*!
     * Clears the policy output vectors. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.clear_policy_outputs.
     */
    rolloutBufferContainer_->clear_policy_outputs();
}

size_t C_RolloutBuffer::size_transitions() {
    /*!
     * Returns the size of Rollout Buffer. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.size_transitions.
     */
    return rolloutBufferContainer_->size_transitions();
}

size_t C_RolloutBuffer::size_policy_outputs() {
    /*!
     * Returns the size of Rollout Buffer. This method is C++ backend for
     * rlpack._C.rollout_buffer.RolloutBuffer.size_policy_outputs.
     */
    return rolloutBufferContainer_->size_policy_outputs();
}

void C_RolloutBuffer::extend_transitions(std::map<std::string, std::vector<torch::Tensor>>& extensionMap) {
    /*!
     * This method extends the transitions by performing gather for transition quantities. This method will throw a
     * runtime error if process group is not defined in Python. If extension map is empty, returns immediately. This
     * method is C++ backend for rlpack._C.rollout_buffer.RolloutBuffer.extend_transitions.
     */
    if (extensionMap[STATES_CURRENT].empty() or
        extensionMap[STATES_NEXT].empty() or
        extensionMap[REWARDS].empty() or
        extensionMap[DONES].empty()) {
        return;
    }
    rolloutBufferContainer_->extend_transitions(extensionMap);
}

void C_RolloutBuffer::set_transitions_iterator(int64_t batchSize) {
    /*!
     * A utility method which is used to set the private attribute dataloader_ with appropriate dataloader. This
     * method creates the MapDataset and wraps it with SequentialSampler to create the dataloader.
     *
     * @param batchSize: The batch size to be used to batchify the transitions.
     */
    auto dataset = RolloutBufferBatch(rolloutBufferContainer_, tensorOptions_);
    auto mapDataset = dataset.map(RolloutBufferBatchTransform(batchSize));
    dataloader_ = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(mapDataset,
                                                                                          batchSize);
}

C_RolloutBuffer::DataLoader& C_RolloutBuffer::get_dataloader_reference() {
    /*!
     * Returns the reference to private attribute dataloader_. Note that dataloader_ is a unique_ptr and hence
     * implicit copy is not allowed. This method must not be called before `set_transitions_iterator` else an
     * uninitialized reference might be returned. This method is used to bind dataloader_ iterator in python via
     * pybind11.
     */
    return dataloader_;
}
