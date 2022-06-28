//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_SRC_DQN_AGENT_TPP_
#define RLPACK_SRC_DQN_AGENT_TPP_

#include "Agent.h"

namespace dqn{
template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::Memory::Memory() = default;

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::Memory::push_back(torch::Tensor &stateCurrent,
                                                     torch::Tensor &stateNext,
                                                     float reward,
                                                     int action,
                                                     int done) {

  stateCurrent_.push_back(stateCurrent);
  stateNext_.push_back(stateNext);

  torch::TensorOptions tensorOptionsForReward = torch::TensorOptions().dtype(torch::kDouble);
  torch::Tensor rewardAsTensor = torch::full({1}, reward, tensorOptionsForReward);
  reward_.push_back(rewardAsTensor);

  torch::TensorOptions tensorOptionsForAction = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor actionAsTensor = torch::full({1}, action, tensorOptionsForAction);
  action_.push_back(actionAsTensor);

  torch::TensorOptions tensorOptionsForDone = torch::TensorOptions().dtype(torch::kDouble);
  torch::Tensor doneAsTensor = torch::full({1}, done, tensorOptionsForDone);
  done_.push_back(doneAsTensor);
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::Memory::clear() {
  stateCurrent_.clear();
  stateNext_.clear();
  reward_.clear();
  action_.clear();
  done_.clear();
}

template<class ModelClass, class Optimizer>
typename Agent<ModelClass, Optimizer>::Memory *Agent<ModelClass, Optimizer>::Memory::at(int index) {
  auto *memory = new Memory();
  memory->stateCurrent_ = {stateCurrent_[index]};
  memory->stateNext_ = {stateNext_[index]};
  memory->reward_ = {reward_[index]};
  memory->action_ = {action_[index]};
  memory->done_ = {done_[index]};

  return memory;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::Memory::push_back(typename Agent<ModelClass, Optimizer>::Memory *memory) {
  stateCurrent_.push_back(memory->stateCurrent_[0]);
  stateNext_.push_back(memory->stateNext_[0]);
  reward_.push_back(memory->reward_[0]);
  action_.push_back(memory->action_[0]);
  done_.push_back(memory->done_[0]);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_current_states() {
  return torch::stack(stateCurrent_, 0);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_next_states() {
  return torch::stack(stateNext_, 0);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_rewards() {
  return torch::stack(reward_, 0);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_actions() {
  return torch::stack(action_, 0);
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::Memory::stack_dones() {
  return torch::stack(done_, 0);
}

template<class ModelClass, class Optimizer>
size_t Agent<ModelClass, Optimizer>::Memory::size() {
  return done_.size();
}

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::Agent(ModelClass &targetModel,
                                    ModelClass &policyModel,
                                    Optimizer &optimizer,
                                    float gamma, float epsilon,
                                    float epsilonDecayRate,
                                    int memoryBufferSize,
                                    int targetModelUpdateRate,
                                    int policyModelUpdateRate,
                                    int numActions,
                                    std::string &savePath) {
  targetModel_ = targetModel;
  policyModel_ = policyModel;
  optimizer_ = optimizer;

  gamma_ = gamma;
  epsilon_ = epsilon;
  epsilonDecayRate_ = epsilonDecayRate;
  memoryBufferSize_ = memoryBufferSize;
  assert(targetModelUpdateRate > policyModelUpdateRate);

  targetModelUpdateRate_ = targetModelUpdateRate;
  policyModelUpdateRate_ = policyModelUpdateRate;
  numActions_ = numActions;

  savePath_ = savePath.append("tmp.pth");
}

template<class ModelClass, class Optimizer>
int Agent<ModelClass, Optimizer>::train(torch::Tensor &stateCurrent,
                                        torch::Tensor &stateNext,
                                        float reward,
                                        int action,
                                        int done) {

  memoryBuffer.push_back(stateCurrent, stateNext, reward, action, done);
  policyModelUpdateCounter += 1;
  targetModelUpdateCounter += 1;

  if (policyModelUpdateCounter == policyModelUpdateRate_ + 1) {
    train_policy_model();
    policyModelUpdateCounter = 0;
  }

  if (targetModelUpdateCounter == targetModelUpdateRate_) {
    update_target_model();
    targetModelUpdateCounter = 0;
  }

  if (memoryBuffer.size() == memoryBufferSize_) {
    clear_memory();
  }

  if (done == 1) {
    decay_epsilon();
  }

  stateCurrent = stateCurrent.unsqueeze(0);
  action = policy(stateCurrent);
  return action;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::train_policy_model() {
  Memory *randomExperiences = load_random_experiences();

  torch::Tensor statesCurrent = randomExperiences->stack_current_states();
  torch::Tensor statesNext = randomExperiences->stack_next_states();
  torch::Tensor rewards = randomExperiences->stack_rewards();
  torch::Tensor actions = randomExperiences->stack_actions();
  torch::Tensor dones = randomExperiences->stack_dones();

  policyModel_->train();

  torch::Tensor tdValue;
  {
    targetModel_->eval();
    torch::NoGradGuard guard;
    torch::Tensor qValuesTarget = targetModel_->forward(statesNext);
    tdValue = temporal_difference(rewards, qValuesTarget, dones);
  }
  torch::Tensor qValuesPolicy = policyModel_->forward(statesCurrent);
  torch::Tensor qValuesPolicyGathered = qValuesPolicy.gather(-1, actions);

  optimizer_->zero_grad();
  torch::Tensor loss = huberLoss_(tdValue.detach(), qValuesPolicyGathered);
  loss.backward();
  optimizer_->step();
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::update_target_model() {
  torch::serialize::OutputArchive outputArchive;
  torch::serialize::InputArchive inputArchive;

  policyModel_->save(outputArchive);
  outputArchive.save_to(savePath_);

  inputArchive.load_from(savePath_);
  targetModel_->load(inputArchive);
}

template<class ModelClass, class Optimizer>
typename Agent<ModelClass, Optimizer>::Memory *Agent<ModelClass, Optimizer>::load_random_experiences() {
  std::vector<int> loadedIndices;
  std::random_device rd;
  std::mt19937 generator(rd());
  int index;

  auto *loadedExperiences = new Memory();

  while (loadedExperiences->size() != policyModelUpdateRate_) {
    std::uniform_int_distribution<int> distribution(0, memoryBuffer.size() - 1);
    index = distribution(generator);

    Memory *memory = memoryBuffer.at(index);
    loadedExperiences->push_back(memory);
  }

  return loadedExperiences;
}

template<class ModelClass, class Optimizer>
torch::Tensor Agent<ModelClass, Optimizer>::temporal_difference(torch::Tensor &rewards,
                                                                torch::Tensor &qValues,
                                                                torch::Tensor &dones) {
  torch::Tensor tdValue = rewards + ((gamma_ * qValues.max()) * (1 - dones));
  return tdValue;
}

template<class ModelClass, class Optimizer>
int Agent<ModelClass, Optimizer>::policy(torch::Tensor &stateCurrent) {
  int action;
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distributionP(0, 1);
  std::uniform_int_distribution<int> distributionAction(0, numActions_ - 1);
  float p = distributionP(generator);

  if (p < epsilon_) {
    action = distributionAction(generator);
  } else {
    {
      policyModel_->eval();
      torch::NoGradGuard guard;
      torch::Tensor qValues = policyModel_->forward(stateCurrent.toType(torch::kDouble));
      torch::Tensor actionTensor = qValues.argmax(-1);

      action = actionTensor.item<int>();
    }
  }

  return action;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::decay_epsilon() {
  epsilon_ *= epsilonDecayRate_;
}

template<class ModelClass, class Optimizer>
void Agent<ModelClass, Optimizer>::clear_memory() {
  memoryBuffer.clear();
}

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::Memory::~Memory() = default;

template<class ModelClass, class Optimizer>
Agent<ModelClass, Optimizer>::~Agent() = default;

}

#endif//RLPACK_SRC_DQN_AGENT_TPP_
