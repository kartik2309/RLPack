//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#include "C_RolloutBuffer.h"

C_RolloutBuffer::C_RolloutBuffer(int64_t bufferSize, std::string &device, std::string &dtype) {
    /*!
     * Class constructor for C_RolloutBuffer. This will allocate necessary memory as per input, and set relevant
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
    rewards_.reserve(bufferSize);
    actionLogProbabilities_.reserve(bufferSize);
    stateCurrentValues_.reserve(bufferSize);
    entropies_.reserve(bufferSize);
}

/*!
 * Default destructor C_GradAccumulator
 */
C_RolloutBuffer::~C_RolloutBuffer() = default;

void C_RolloutBuffer::insert(std::map<std::string, torch::Tensor> &inputMap) {
    /*!
     * Insertion method to insert quantities into Rollout Buffer.
     *
     * @param inputMap: The input map of tensors with quantities to insert into the rollout buffer. The map must
     *  contain the following keys:
     *  - reward: The reward obtained by the agent.
     *  - action_log_probability: The log probability of the sampled action in the given distribution.
     *  - state_current_value: Current state value.
     *  - entropy: Current entropy of the distribution.
     *
     * For more information, please refer rlpack._C.rollout_buffer.RolloutBuffer and
     * rlpack.actor_critic.base.ActorCriticAgent
     */
    if (rewards_.size() >= bufferSize_) {
        std::string errorMessage = "With given buffer size of " +
                                   std::to_string(bufferSize_) +
                                   " attempted to add more elements";
        throw std::out_of_range(errorMessage.c_str());
    }
    rewards_.push_back(inputMap["reward"]);
    actionLogProbabilities_.push_back(inputMap["action_log_probability"]);
    stateCurrentValues_.push_back(inputMap["state_current_value"]);
    entropies_.push_back(inputMap["entropy"]);
}
std::map<std::string, torch::Tensor> C_RolloutBuffer::compute_returns(float_t gamma) {
    /*!
     * Computes the returns with accumulated rewards.
     *
     * @param gamma: The discounting factor to be used for computing rewards.
     * @return A map of PyTorch tensor of returns, keyed "returns"
     */
    auto totalRewards = rewards_.size();
    std::vector<torch::Tensor> returns_(totalRewards);
    auto r_ = torch::zeros({}, tensorOptions_);
    for (uint64_t index = totalRewards - 1; index != -1; index--) {
        r_ = rewards_[index] + gamma * r_;
        returns_[index] = r_;
    }
    auto returns = torch::stack(returns_).to(tensorOptions_);
    return {{"returns", returns}};
}

void C_RolloutBuffer::clear() {
    /*!
     * Clears the relevant vectors
     */
    rewards_.clear();
    actionLogProbabilities_.clear();
    stateCurrentValues_.clear();
    entropies_.clear();
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_rewards() {
    /*!
     * Stacks the accumulated rewards and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of rewards, keyed "rewards".
     */
    return {{"rewards", torch::stack(rewards_).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_action_log_probabilities() {
    /*!
     * Stacks the accumulated action log probabilities and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of log of action probabilities, keyed "action_log_probabilities".
     */
    return {{"action_log_probabilities", torch::stack(actionLogProbabilities_).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_state_current_values() {
    /*!
     * Stacks the accumulated current state values and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of current states, keyed "state_current_values".
     */
    return {{"state_current_values", torch::stack(stateCurrentValues_).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_entropies() {
    /*!
     * Stacks the accumulated current entropies and moves them to correct tensor options (for device and datatype).
     *
     * @return A map of PyTorch tensor of entropies, keyed "entropies".
     */
    return {{"entropies", torch::stack(entropies_).to(tensorOptions_)}};
}
