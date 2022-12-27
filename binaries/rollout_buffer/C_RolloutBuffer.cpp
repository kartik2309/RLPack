//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#include "C_RolloutBuffer.h"

C_RolloutBuffer::C_RolloutBuffer(int64_t bufferSize, std::string &device, std::string &dtype) {
    bufferSize_ = bufferSize;
    device_ = Maps::deviceMap[device];
    dtype_ = Maps::dTypeMap[dtype];
    tensorOptions_ = torch::TensorOptions().device(device_).dtype(dtype_);
    rewards_.reserve(bufferSize);
    actionLogProbabilities_.reserve(bufferSize);
    stateCurrentValues_.reserve(bufferSize);
    entropies_.reserve(bufferSize);
}

C_RolloutBuffer::~C_RolloutBuffer() = default;

void C_RolloutBuffer::insert(std::map<std::string, torch::Tensor> &inputMap) {
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
    rewards_.clear();
    actionLogProbabilities_.clear();
    stateCurrentValues_.clear();
    entropies_.clear();
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_rewards() {
    return {{"rewards", torch::stack(rewards_).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_action_probabilities() {
    return {{"action_log_probabilities", torch::stack(actionLogProbabilities_).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_state_current_values() {
    return {{"state_current_values", torch::stack(stateCurrentValues_).to(tensorOptions_)}};
}

std::map<std::string, torch::Tensor> C_RolloutBuffer::get_stacked_entropies() {
    return {{"entropies", torch::stack(entropies_).to(tensorOptions_)}};
}
