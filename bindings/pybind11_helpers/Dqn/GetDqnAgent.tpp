//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_
#define RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_

#include "GetDqnAgent.h"

GetDqnAgent::GetDqnAgent(pybind11::str &modelName,
                         pybind11::dict &modelArgs,
                         pybind11::dict &activationArgs,
                         pybind11::dict &agentArgs,
                         pybind11::dict &optimizerArgs) {
  modelName_ = modelName.cast<std::string>();
  modelArgs_ = modelArgs;
  activationArgsRetrieved_ = activationArgs;
  agentArgs_ = agentArgs;
  optimizerArgsRetrieved_ = optimizerArgs;

  agent_ = get_agent();
}

std::vector<dqn::DqnImpl *> GetDqnAgent::get_dqn_model() {
  auto sequenceLength = modelArgs_["sequence_length"].cast<int64_t>();
  auto channels = modelArgs_["channels"].cast<std::vector<int64_t>>();
  auto kernelSizesConv = modelArgs_["kernel_sizes_conv"].cast<std::vector<int64_t>>();
  auto strideSizesConv = modelArgs_["strides_sizes_conv"].cast<std::vector<int64_t>>();
  auto dilationSizesConv = modelArgs_["dilation_sizes_conv"].cast<std::vector<int64_t>>();
  auto kernelSizesPool = modelArgs_["kernel_sizes_pool"].cast<std::optional<std::vector<int64_t>>>();
  auto strideSizesPool = modelArgs_["strides_sizes_pool"].cast<std::optional<std::vector<int64_t>>>();
  auto dilationSizesPool = modelArgs_["dilation_sizes_pool"].cast<std::optional<std::vector<int64_t>>>();
  auto activation = modelArgs_["activation"].cast<std::string>();
  auto dropout = modelArgs_["dropout"].cast<float_t>();
  auto numActions = modelArgs_["num_actions"].cast<int64_t>();
  auto usePadding = modelArgs_["use_padding"].cast<bool>();

  auto modelCode = dqnModels_[modelName_];
  auto activationCode = activations_[activation];

  dqn::DqnImpl *targetModel;
  dqn::DqnImpl *policyModel;

  switch (modelCode) {
    case 0: {
      // When the model is Dqn1d
      switch (activationCode) {

        case 0: {
          // When activation is ReLU
          torch::nn::ReLU relu = torch::nn::ReLU();
          targetModel = new dqn::Dqn1d<torch::nn::ReLU>(sequenceLength,
                                                        channels,
                                                        kernelSizesConv,
                                                        strideSizesConv,
                                                        dilationSizesConv,
                                                        numActions,
                                                        kernelSizesPool,
                                                        strideSizesPool,
                                                        dilationSizesPool,
                                                        relu,
                                                        dropout,
                                                        usePadding);
          policyModel = new dqn::Dqn1d<torch::nn::ReLU>(sequenceLength,
                                                        channels,
                                                        kernelSizesConv,
                                                        strideSizesConv,
                                                        dilationSizesConv,
                                                        numActions,
                                                        kernelSizesPool,
                                                        strideSizesPool,
                                                        dilationSizesPool,
                                                        relu,
                                                        dropout,
                                                        usePadding);
          targetModel->to_double();
          policyModel->to_double();

          std::vector<dqn::DqnImpl *> dqnModels = {targetModel, policyModel};
          return dqnModels;
        }
        case 1: {
          // When activation is LeakyReLU
          torch::nn::LeakyReLUOptions leakyReluOptions = torch::nn::LeakyReLUOptions();
          auto leakyReluArgs = activationArgs_["leaky_relu"];

          if (activationArgsRetrieved_.contains(leakyReluArgs.at(0))) {
            auto negativeSlope = activationArgsRetrieved_[leakyReluArgs.at(0).c_str()].cast<double>();
            leakyReluOptions.negative_slope(negativeSlope);
          }

          torch::nn::LeakyReLU leakyRelu = torch::nn::LeakyReLU(leakyReluOptions);
          targetModel = new dqn::Dqn1d<torch::nn::LeakyReLU>(sequenceLength,
                                                             channels,
                                                             kernelSizesConv,
                                                             strideSizesConv,
                                                             dilationSizesConv,
                                                             numActions,
                                                             kernelSizesPool,
                                                             strideSizesPool,
                                                             dilationSizesPool,
                                                             leakyRelu,
                                                             dropout,
                                                             usePadding);
          policyModel = new dqn::Dqn1d<torch::nn::LeakyReLU>(sequenceLength,
                                                             channels,
                                                             kernelSizesConv,
                                                             strideSizesConv,
                                                             dilationSizesConv,
                                                             numActions,
                                                             kernelSizesPool,
                                                             strideSizesPool,
                                                             dilationSizesPool,
                                                             leakyRelu,
                                                             dropout,
                                                             usePadding);

          targetModel->to_double();
          policyModel->to_double();

          std::vector<dqn::DqnImpl *> dqnModels = {targetModel, policyModel};
          return dqnModels;
        }
        case 3: {
          // When activation is sigmoid.
          torch::nn::Sigmoid sigmoid = torch::nn::Sigmoid();
          targetModel = new dqn::Dqn1d<torch::nn::Sigmoid>(sequenceLength,
                                                           channels,
                                                           kernelSizesConv,
                                                           strideSizesConv,
                                                           dilationSizesConv,
                                                           numActions,
                                                           kernelSizesPool,
                                                           strideSizesPool,
                                                           dilationSizesPool,
                                                           sigmoid,
                                                           dropout,
                                                           usePadding);
          policyModel = new dqn::Dqn1d<torch::nn::Sigmoid>(sequenceLength,
                                                           channels,
                                                           kernelSizesConv,
                                                           strideSizesConv,
                                                           dilationSizesConv,
                                                           numActions,
                                                           kernelSizesPool,
                                                           strideSizesPool,
                                                           dilationSizesPool,
                                                           sigmoid,
                                                           dropout,
                                                           usePadding);
          targetModel->to_double();
          policyModel->to_double();

          std::vector<dqn::DqnImpl *> dqnModels = {targetModel, policyModel};
          return dqnModels;
        }
        default:
          break;
      }
    }
    default:
      throw std::runtime_error("Invalid combination of values encountered for model_name and activation");
  }
}

AgentImpl *GetDqnAgent::get_agent() {
  auto gamma = agentArgs_["gamma"].cast<float>();
  auto epsilon = agentArgs_["epsilon"].cast<float>();
  auto epsilonDecayRate = agentArgs_["epsilon_decay_rate"].cast<float>();
  auto memoryBufferSize = agentArgs_["memory_buffer_size"].cast<int>();
  auto targetModelUpdateRate = agentArgs_["target_model_update_rate"].cast<int>();
  auto policyModelUpdateRate = agentArgs_["policy_model_update_rate"].cast<int>();
  auto numActions = agentArgs_["num_actions"].cast<int>();
  auto savePath = agentArgs_["save_path"].cast<std::string>();

  auto optimizerName = optimizerArgsRetrieved_["optimizer"].cast<std::string>();
  auto optimizerCode = optimizers_[optimizerName];
  auto models = get_dqn_model();
  switch (optimizerCode) {
    case 0: {

      // When optimizer is Adam
      auto optimizersArgs = optimizersArgs_[optimizerName];
      auto lr = optimizerArgsRetrieved_[optimizersArgs.at(0).c_str()].cast<pybind11::float_>().cast<float_t>();
      torch::optim::AdamOptions adamOptions = torch::optim::AdamOptions(lr);

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(1).c_str())) {
        adamOptions.betas(optimizerArgsRetrieved_[optimizersArgs.at(1).c_str()].cast<std::tuple<double_t, double_t>>());
      }

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(2).c_str())) {
        adamOptions.eps(optimizerArgsRetrieved_[optimizersArgs.at(2).c_str()].cast<double_t>());
      }

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(3).c_str())) {
        adamOptions.weight_decay(optimizerArgsRetrieved_[optimizersArgs.at(3).c_str()].cast<double_t>());
      }

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(4).c_str())) {
        adamOptions.amsgrad(optimizerArgsRetrieved_[optimizersArgs.at(4).c_str()].cast<bool>());
      }

      auto *adamOptimizer = new torch::optim::Adam(models.at(1)->parameters(), adamOptions);
      auto *agent = new dqn::Agent<dqn::DqnImpl *, torch::optim::Adam *>(models.at(0),
                                                                         models.at(1),
                                                                         adamOptimizer,
                                                                         gamma,
                                                                         epsilon,
                                                                         epsilonDecayRate,
                                                                         memoryBufferSize,
                                                                         targetModelUpdateRate,
                                                                         policyModelUpdateRate,
                                                                         numActions,
                                                                         savePath);
      return agent;
    }
    case 1: {

      // When optimizer is RMSProp
      auto optimizersArgs = optimizersArgs_[optimizerName];
      auto lr = optimizerArgsRetrieved_[optimizersArgs.at(0).c_str()].cast<float_t>();
      torch::optim::RMSpropOptions rmsPropOptions = torch::optim::RMSpropOptions(lr);

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(1).c_str())) {
        rmsPropOptions.alpha(optimizerArgsRetrieved_[optimizersArgs.at(1).c_str()].cast<double_t>());
      }

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(2).c_str())) {
        rmsPropOptions.eps(optimizerArgsRetrieved_[optimizersArgs.at(2).c_str()].cast<double_t>());
      }

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(3).c_str())) {
        rmsPropOptions.weight_decay(optimizerArgsRetrieved_[optimizersArgs.at(3).c_str()].cast<double_t>());
      }

      if (optimizerArgsRetrieved_.contains(optimizersArgs.at(4).c_str())) {
        rmsPropOptions.momentum(optimizerArgsRetrieved_[optimizersArgs.at(4).c_str()].cast<double_t>());
      }

      auto *rmsPropOptimizer = new torch::optim::RMSprop(models.at(1)->parameters(), rmsPropOptions);
      auto *agent = new dqn::Agent<dqn::DqnImpl *, torch::optim::RMSprop *>(models.at(0),
                                                                            models.at(1),
                                                                            rmsPropOptimizer,
                                                                            gamma,
                                                                            epsilon,
                                                                            epsilonDecayRate,
                                                                            memoryBufferSize,
                                                                            targetModelUpdateRate,
                                                                            policyModelUpdateRate,
                                                                            numActions,
                                                                            savePath);
      return agent;
    }
    default:
      throw std::runtime_error("Invalid Optimizer name passed!");
  }
}

int GetDqnAgent::train(pybind11::array_t<double_t> &stateCurrent,
                       pybind11::array_t<double_t> &stateNext,
                       pybind11::float_ &reward,
                       pybind11::int_ &action,
                       pybind11::bool_ &done,
                       pybind11::tuple &stateCurrentShape,
                       pybind11::tuple &stateNextShape) {

  // Convert pybind::array (numpy arrays) to torch::Tensor.
  auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape);
  auto stateNextTensor_ = pybind_array_to_torch_tensor(stateNext, stateNextShape);

  // Convert other arguments to native C++ types.
  auto reward_ = reward.cast<float>();
  auto action_ = action.cast<int>();
  auto done_ = done.cast<int>();

  action_ = agent_->train(stateCurrentTensor_, stateNextTensor_, reward_, action_, done_);
  return action_;
}

int GetDqnAgent::policy(pybind11::array_t<double_t> &stateCurrent, pybind11::tuple &stateCurrentShape) {
  auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape);
  auto action = agent_->policy(stateCurrentTensor_);
  return action;
}

torch::Tensor GetDqnAgent::pybind_array_to_torch_tensor(pybind11::array_t<double_t> &array, pybind11::tuple &shape) {
  std::vector<int64_t> shapeVector;
  for (auto &shape_ : shape) {
    shapeVector.push_back(shape_.cast<int>());
  }

  auto tensorOptions = torch::TensorOptions().dtype(torch::kDouble);
  auto tensor = torch::zeros(shapeVector, tensorOptions);
  memmove(tensor.data_ptr(), array.data(), tensor.nbytes());

  return tensor;
}

GetDqnAgent::~GetDqnAgent() = default;

#endif//RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_