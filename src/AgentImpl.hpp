//
// Created by Kartik Rajeshwaran on 2022-06-20.
//

#ifndef RLPACK_SRC_AGENTIMPL_HPP_
#define RLPACK_SRC_AGENTIMPL_HPP_

#include <torch/torch.h>

class AgentImpl{

 public:
  AgentImpl();
  virtual int train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done);
  virtual int policy(torch::Tensor &stateCurrent);
  virtual void save(std::string &path);
};

int AgentImpl::train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward, int action, int done){
  return 0;
}

int AgentImpl::policy(torch::Tensor &stateCurrent) {
  return 0;
}

void AgentImpl::save(std::string &path) {}

AgentImpl::AgentImpl() = default;

#endif//RLPACK_SRC_AGENTIMPL_HPP_
