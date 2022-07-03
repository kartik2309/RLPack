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

#include "../../../src/AgentBase.hpp"
#include "../../../src/Dqn/Agent.tpp"
#include "../../../src/Dqn/Dcqn1d/Dcqn1d.tpp"
#include "../../../src/Dqn/Dlqn1d/Dlqn1d.tpp"
#include "../../../src/Dqn/DqnBase.hpp"

class GetDqnAgent {
public:
    GetDqnAgent(pybind11::str &modelName, pybind11::dict &modelArgs, pybind11::dict &activationArgs,
                pybind11::dict &agentArgs, pybind11::dict &optimizerArgs, pybind11::str &device);

    ~GetDqnAgent();

    int train(pybind11::array_t<float_t> &stateCurrent, pybind11::array_t<float_t> &stateNext,
              pybind11::float_ &reward, pybind11::int_ &action, pybind11::bool_ &done,
              pybind11::tuple &stateCurrentShape, pybind11::tuple &stateNextShape);

    int policy(pybind11::array_t<float_t> &stateCurrent, pybind11::tuple &stateCurrentShape);

    void save();

private:
    std::string modelName_;
    pybind11::kwargs modelArgs_;
    pybind11::kwargs activationArgsRetrieved_;
    pybind11::kwargs agentArgs_;
    pybind11::kwargs optimizerArgsRetrieved_;
    AgentBase *agent_;
    torch::DeviceType device_;
    torch::ScalarType dataType_;

    std::map<std::string, int> dqnModels_ = {{"dcqn1d", 0},
                                             {"dlqn1d", 1}};
    std::map<std::string, int> activations_ = {{"relu",       0},
                                               {"leaky_relu", 1},
                                               {"sigmoid",    2}};
    std::map<std::string, std::vector<std::string>> activationArgs_ = {
            {"relu",       {}},
            {"leaky_relu", {"negative_slope"}}};

    std::map<std::string, int> optimizers_ = {{"adam",    0},
                                              {"rmsprop", 1},
                                              {"sgd",     2}};
    std::map<std::string, std::vector<std::string>> optimizersArgs_ = {
            {"adam",    {"lr", "betas",    "eps",       "weight_decay", "amsgrad"}},
            {"rmsprop", {"lr", "alpha",    "eps",       "weight_decay", "momentum"}},
            {"sgd",     {"lr", "momentum", "dampening", "weight_decay", "nesterov"}}};

    torch::Tensor pybind_array_to_torch_tensor(pybind11::array_t<float_t> &array, pybind11::tuple &shape);

    AgentBase *get_agent();

    std::vector<std::shared_ptr<dqn::DqnBase>> get_dqn_model();

    std::vector<std::shared_ptr<dqn::DqnBase>>
    get_relu_dcqn1d_model(int32_t sequenceLength, std::vector<int32_t> &channels,
                          std::vector<int32_t> &kernelSizesConv,
                          std::vector<int32_t> &strideSizesConv,
                          std::vector<int32_t> &dilationSizesConv,
                          std::optional<std::vector<int32_t>> &kernelSizesPool,
                          std::optional<std::vector<int32_t>> &strideSizesPool,
                          std::optional<std::vector<int32_t>> &dilationSizesPool,
                          float_t dropout, int32_t numActions, bool usePadding);

    std::vector<std::shared_ptr<dqn::DqnBase>>
    get_relu_dlqn1d_model(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                          float_t dropout);

    std::vector<std::shared_ptr<dqn::DqnBase>>
    get_leaky_relu_dcqn1d_model(int32_t sequenceLength, std::vector<int32_t> &channels,
                                std::vector<int32_t> &kernelSizesConv,
                                std::vector<int32_t> &strideSizesConv,
                                std::vector<int32_t> &dilationSizesConv,
                                std::optional<std::vector<int32_t>> &kernelSizesPool,
                                std::optional<std::vector<int32_t>> &strideSizesPool,
                                std::optional<std::vector<int32_t>> &dilationSizesPool,
                                float_t dropout, int32_t numActions, bool usePadding);

    std::vector<std::shared_ptr<dqn::DqnBase>>
    get_leaky_relu_dlqn1d_model(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                          float_t dropout);

    std::vector<std::shared_ptr<dqn::DqnBase>>
    get_sigmoid_dcqn1d_model(int32_t sequenceLength, std::vector<int32_t> &channels,
                             std::vector<int32_t> &kernelSizesConv,
                             std::vector<int32_t> &strideSizesConv,
                             std::vector<int32_t> &dilationSizesConv,
                             std::optional<std::vector<int32_t>> &kernelSizesPool,
                             std::optional<std::vector<int32_t>> &strideSizesPool,
                             std::optional<std::vector<int32_t>> &dilationSizesPool,
                             float_t dropout, int32_t numActions, bool usePadding);

    std::vector<std::shared_ptr<dqn::DqnBase>>
    get_sigmoid_dlqn1d_model(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                          float_t dropout);

    AgentBase *
    get_adam_optim_agent(std::shared_ptr<dqn::DqnBase> &targetModel, std::shared_ptr<dqn::DqnBase> &policyModel,
                         float_t gamma, float_t epsilon, float_t epsilonDecayRate, int32_t memoryBufferSize,
                         int32_t targetModelUpdateRate, int32_t policyModelUpdateRate, int32_t numActions,
                         std::string &savePath, int32_t applyNorm, int32_t applyNormTo);

    AgentBase *
    get_rmsprop_optim_agent(std::shared_ptr<dqn::DqnBase> &targetModel, std::shared_ptr<dqn::DqnBase> &policyModel,
                            float_t gamma, float_t epsilon, float_t epsilonDecayRate, int32_t memoryBufferSize,
                            int32_t targetModelUpdateRate, int32_t policyModelUpdateRate, int32_t numActions,
                            std::string &savePath, int32_t applyNorm, int32_t applyNormTo);

    static torch::DeviceType get_device(std::string &device);

    static torch::ScalarType get_data_type(std::string &device);
};

#endif//RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_H_
