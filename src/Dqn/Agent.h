//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_AGENT_H_
#define RLPACK_DQN_AGENT_H_

#include <random>
#include <torch/torch.h>

#include "../AgentImpl.hpp"
#include "Dqn1d/Dqn1d.h"

namespace dqn {
template<class ModelClass, class Optimizer>
class Agent : public AgentImpl {

  ModelClass targetModel_;
  ModelClass policyModel_;
  Optimizer optimizer_;
  float gamma_;
  float epsilon_;
  float epsilonDecayRate_;
  int memoryBufferSize_;
  int targetModelUpdateRate_;
  int policyModelUpdateRate_;
  int numActions_;
  std::string savePath_;

  torch::nn::HuberLoss huberLoss_;

  struct Memory {
   private:
    std::vector<torch::Tensor> stateCurrent_;
    std::vector<torch::Tensor> stateNext_;
    std::vector<torch::Tensor> reward_;
    std::vector<torch::Tensor> action_;
    std::vector<torch::Tensor> done_;

   public:
    Memory();
    ~Memory();
    void push_back(torch::Tensor &stateCurrent,
                   torch::Tensor &stateNext,
                   float reward,
                   int action,
                   int done);
    void push_back(Memory *memory);
    torch::Tensor stack_current_states();
    torch::Tensor stack_next_states();
    torch::Tensor stack_rewards();
    torch::Tensor stack_actions();
    torch::Tensor stack_dones();
    void clear();
    size_t size();
    Memory *at(int index);
  } memoryBuffer;

  int targetModelUpdateCounter = 0;
  int policyModelUpdateCounter = 0;

  void train_policy_model();
  void update_target_model();
  typename Agent<ModelClass, Optimizer>::Memory *load_random_experiences();
  torch::Tensor temporal_difference(torch::Tensor &rewards, torch::Tensor &qValues, torch::Tensor &dones);
  void decay_epsilon();
  void clear_memory();

 public:
  Agent(ModelClass &targetModel,
        ModelClass &policyModel,
        Optimizer &optimizer,
        float gamma,
        float epsilon,
        float epsilonDecayRate,
        int memoryBufferSize,
        int targetModelUpdateRate,
        int policyModelUpdateRate,
        int numActions,
        std::string &savePath);
  ~Agent();

  int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done) override;
  int policy(torch::Tensor &stateCurrent) override;
};

}// namespace dqn
#endif//RLPACK_DQN_AGENT_H_
