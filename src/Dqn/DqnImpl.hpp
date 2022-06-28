//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_SRC_DQN_DQNIMPL_HPP_
#define RLPACK_SRC_DQN_DQNIMPL_HPP_
#include <torch/torch.h>

namespace dqn {
class DqnImpl : public torch::nn::Module {

 public:
  virtual torch::Tensor forward(torch::Tensor x);
  virtual void to_double();
  ~DqnImpl() override;
};

torch::Tensor DqnImpl::forward(torch::Tensor x) {
  return x;
}

void DqnImpl::to_double() {}

DqnImpl::~DqnImpl() = default;
}// namespace dqn

#endif//RLPACK_SRC_DQN_DQNIMPL_HPP_
