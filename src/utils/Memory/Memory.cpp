//
// Created by Kartik Rajeshwaran on 2022-07-05.
//

#include "Memory.h"

Memory::Memory() = default;


Memory::Memory(int32_t bufferSize) {
    reserve(bufferSize);
}

void Memory::push_back(torch::Tensor &stateCurrent, torch::Tensor &stateNext,
                       float reward, int action, int done) {

    stateCurrent_.push_back(stateCurrent);
    stateNext_.push_back(stateNext);

    torch::TensorOptions tensorOptionsForReward = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor rewardAsTensor = torch::full({1}, reward, tensorOptionsForReward);
    reward_.push_back(rewardAsTensor);

    torch::TensorOptions tensorOptionsForAction = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor actionAsTensor = torch::full({1}, action, tensorOptionsForAction);
    action_.push_back(actionAsTensor);

    torch::TensorOptions tensorOptionsForDone = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor doneAsTensor = torch::full({1}, done, tensorOptionsForDone);
    done_.push_back(doneAsTensor);
}

void Memory::clear() {
    stateCurrent_.clear();
    stateNext_.clear();
    reward_.clear();
    action_.clear();
    done_.clear();
}

void Memory::at(Memory *memory, int index) {
    memory->stateCurrent_.push_back(stateCurrent_[index]);
    memory->stateNext_.push_back(stateNext_[index]);
    memory->reward_.push_back(reward_[index]);
    memory->action_.push_back(action_[index]);
    memory->done_.push_back(done_[index]);
}

torch::Tensor Memory::stack_current_states() {
    torch::Tensor stateCurrentStacked = torch::stack(stateCurrent_, 0);
    return stateCurrentStacked;
}

torch::Tensor Memory::stack_next_states() {
    torch::Tensor stateNextStacked = torch::stack(stateNext_, 0);
    return stateNextStacked;
}

torch::Tensor Memory::stack_rewards() {
    torch::Tensor rewardsStacked = torch::stack(reward_, 0);
    return rewardsStacked;
}

torch::Tensor Memory::stack_actions() {
    torch::Tensor actionStacked = torch::stack(action_, 0);
    return actionStacked;
}

torch::Tensor Memory::stack_dones() {
    return torch::stack(done_, 0);
}

size_t Memory::size() {
    return done_.size();
}


void Memory::reserve(int32_t bufferSize) {
    stateCurrent_.reserve(bufferSize);
    stateNext_.reserve(bufferSize);
    reward_.reserve(bufferSize);
    action_.reserve(bufferSize);
    done_.reserve(bufferSize);
}

Memory::~Memory() = default;
