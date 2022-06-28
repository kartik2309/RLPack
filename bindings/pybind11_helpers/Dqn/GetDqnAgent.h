//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_H_
#define RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "../../../src/AgentImpl.hpp"
#include "../../../src/Dqn/Agent.tpp"
#include "../../../src/Dqn/Dqn1d/Dqn1d.tpp"
#include "../../../src/Dqn/DqnImpl.hpp"

class GetDqnAgent {
  std::string modelName_;
  pybind11::kwargs modelArgs_;
  pybind11::kwargs activationArgsRetrieved_;
  pybind11::kwargs agentArgs_;
  pybind11::kwargs optimizerArgsRetrieved_;
  AgentImpl *agent_;

  std::map<std::string, int> dqnModels_ = {{"dqn1d", 0}, {"dqn2d", 1}};
  std::map<std::string, int> activations_ = {{"relu", 0}, {"leaky_relu", 1}, {"sigmoid", 2}};
  std::map<std::string, std::vector<std::string>> activationArgs_ = {
    {"relu", {}},
    {"leaky_relu", {"negative_slope"}}};

  std::map<std::string, int> optimizers_ = {{"adam", 0}, {"rmsprop", 1}, {"sgd", 2}};
  std::map<std::string, std::vector<std::string>> optimizersArgs_ = {
    {"adam", {"lr", "betas", "eps", "weight_decay", "amsgrad"}},
    {"rmsprop", {"lr", "alpha", "eps", "weight_decay", "momentum"}},
    {"sgd", {"lr", "momentum", "dampening", "weight_decay", "nesterov"}}};

  static torch::Tensor pybind_array_to_torch_tensor(pybind11::array_t<double_t> &array, pybind11::tuple &shape);

  std::vector<dqn::DqnImpl *> get_dqn_model();

 public:
  GetDqnAgent(pybind11::str &modelName,
              pybind11::dict &modelArgs,
              pybind11::dict &activationArgs,
              pybind11::dict &agentArgs,
              pybind11::dict &optimizerArgs);
  ~GetDqnAgent();

  AgentImpl *get_agent();
  int train(pybind11::array_t<double_t> &stateCurrent,
            pybind11::array_t<double_t> &stateNext,
            pybind11::float_ &reward,
            pybind11::int_ &action,
            pybind11::bool_ &done,
            pybind11::tuple &stateCurrentShape,
            pybind11::tuple &stateNextShape);
  int policy(pybind11::array_t<double_t> &stateCurrent, pybind11::tuple &stateCurrentShape);
};

#endif//RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_H_
