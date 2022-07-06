//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_
#define RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_

#include "GetDqnAgent.h"


GetDqnAgent::GetDqnAgent(pybind11::str &modelName, pybind11::dict &modelArgs, pybind11::dict &activationArgs,
                         pybind11::dict &agentArgs, pybind11::dict &optimizerArgs, pybind11::str &device) {
    modelName_ = modelName.cast<std::string>();
    modelArgs_ = modelArgs;
    activationArgsRetrieved_ = activationArgs;
    agentArgs_ = agentArgs;
    optimizerArgsRetrieved_ = optimizerArgs;

    auto deviceName = device.cast<std::string>();
    device_ = get_device(deviceName);
    dataType_ = get_data_type(deviceName);

    agent_ = get_agent();

}

int GetDqnAgent::train(pybind11::array_t<float_t> &stateCurrent, pybind11::array_t<float_t> &stateNext,
                       pybind11::float_ &reward, pybind11::int_ &action, pybind11::bool_ &done,
                       pybind11::tuple &stateCurrentShape, pybind11::tuple &stateNextShape) {

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

int GetDqnAgent::policy(pybind11::array_t<float_t> &stateCurrent, pybind11::tuple &stateCurrentShape) {
    auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape);
    auto action = agent_->policy(stateCurrentTensor_);
    return action;
}

void GetDqnAgent::save() {
    agent_->save();
}

void GetDqnAgent::load() {
    agent_->load();
}

std::vector<std::shared_ptr<dqn::DqnBase>> GetDqnAgent::get_dqn_model() {

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arguments specific to Dcqn Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto channels = modelArgs_.contains("channels")
                    ? modelArgs_["channels"].cast<std::vector<int32_t>>()
                    : std::optional<std::vector<int32_t>>();
    auto kernelSizesConv = modelArgs_.contains("kernel_sizes_conv")
                           ? modelArgs_["kernel_sizes_conv"].cast<std::optional<std::vector<int32_t>>>()
                           : std::optional<std::vector<int32_t>>();
    auto strideSizesConv = modelArgs_.contains("strides_sizes_conv")
                           ? modelArgs_["strides_sizes_conv"].cast<std::optional<std::vector<int32_t>>>()
                           : std::optional<std::vector<int32_t>>();
    auto dilationSizesConv = modelArgs_.contains("dilation_sizes_conv")
                             ? modelArgs_["dilation_sizes_conv"].cast<std::optional<std::vector<int32_t>>>()
                             : std::optional<std::vector<int32_t>>();
    auto kernelSizesPool = modelArgs_.contains("kernel_sizes_pool")
                           ? modelArgs_["kernel_sizes_pool"].cast<std::optional<std::vector<int32_t>>>()
                           : std::optional<std::vector<int32_t>>();
    auto strideSizesPool = modelArgs_.contains("strides_sizes_pool")
                           ? modelArgs_["strides_sizes_pool"].cast<std::optional<std::vector<int32_t>>>()
                           : std::optional<std::vector<int32_t>>();
    auto dilationSizesPool = modelArgs_.contains("dilation_sizes_pool")
                             ? modelArgs_["dilation_sizes_pool"].cast<std::optional<std::vector<int32_t>>>()
                             : std::optional<std::vector<int32_t>>();
    auto usePadding = modelArgs_.contains("use_padding")
                      ? modelArgs_["use_padding"].cast<std::optional<bool>>()
                      : std::optional<bool>();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arguments specific to Dlqn Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto hiddenSizes = modelArgs_.contains("hidden_sizes")
                       ? modelArgs_["hidden_sizes"].cast<std::optional<std::vector<int32_t>>>()
                       : std::optional<std::vector<int32_t>>();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Common Arguments for Dcqn1d and Dlqn1d Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto sequenceLength = modelArgs_.contains("sequence_length")
                          ? modelArgs_["sequence_length"].cast<std::optional<int32_t>>()
                          : std::optional<int32_t>();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Common Arguments for all Dqn Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto activation = modelArgs_["activation"].cast<std::string>();
    auto dropout = modelArgs_["dropout"].cast<float_t>();
    auto numActions = modelArgs_["num_actions"].cast<int32_t>();

    auto modelCode = dqnModels_[modelName_];
    auto activationCode = activations_[activation];

    switch (modelCode) {
        case 0: {
            // When the model is Dcqn1d
            switch (activationCode) {

                case 0: {
                    return get_relu_dcqn1d_model(sequenceLength.value(), channels.value(), kernelSizesConv.value(),
                                                 strideSizesConv.value(), dilationSizesConv.value(), kernelSizesPool,
                                                 strideSizesPool, dilationSizesPool, dropout, numActions,
                                                 usePadding.value());
                }
                case 1: {
                    return get_leaky_relu_dcqn1d_model(sequenceLength.value(), channels.value(),
                                                       kernelSizesConv.value(),
                                                       strideSizesConv.value(), dilationSizesConv.value(),
                                                       kernelSizesPool,
                                                       strideSizesPool, dilationSizesPool, dropout, numActions,
                                                       usePadding.value());
                }
                case 3: {
                    return get_sigmoid_dcqn1d_model(sequenceLength.value(), channels.value(), kernelSizesConv.value(),
                                                    strideSizesConv.value(), dilationSizesConv.value(), kernelSizesPool,
                                                    strideSizesPool, dilationSizesPool, dropout, numActions,
                                                    usePadding.value());
                }
                default:
                    break;
            }
        }
        case 1: {

            // When the model is Dlqn1d
            switch (activationCode) {

                case 0: {
                    return get_relu_dlqn1d_model(sequenceLength.value(), hiddenSizes.value(), numActions, dropout);
                }
                case 1: {
                    return get_leaky_relu_dlqn1d_model(sequenceLength.value(), hiddenSizes.value(), numActions,
                                                       dropout);
                }
                case 3: {
                    return get_sigmoid_dlqn1d_model(sequenceLength.value(), hiddenSizes.value(), numActions, dropout);
                }
                default:
                    break;
            }
        }
        default:
            throw std::runtime_error("Invalid combination of values encountered for model_name and activation");
    }
}

AgentBase *GetDqnAgent::get_agent() {
    auto gamma = agentArgs_["gamma"].cast<float_t>();
    auto epsilon = agentArgs_["epsilon"].cast<float_t>();
    auto minEpsilon = agentArgs_["min_epsilon"].cast<float_t>();
    auto epsilonDecayRate = agentArgs_["epsilon_decay_rate"].cast<float_t>();
    auto epsilonDecayFrequency = agentArgs_["epsilon_decay_frequency"].cast<int32_t>();
    auto memoryBufferSize = agentArgs_["memory_buffer_size"].cast<int32_t>();
    auto targetModelUpdateRate = agentArgs_["target_model_update_rate"].cast<int32_t>();
    auto policyModelUpdateRate = agentArgs_["policy_model_update_rate"].cast<int32_t>();
    auto batchSize = agentArgs_["batch_size"].cast<int32_t>();
    auto numActions = agentArgs_["num_actions"].cast<int32_t>();
    auto savePath = agentArgs_["save_path"].cast<std::string>();
    auto tau = agentArgs_.contains("tau") ? agentArgs_["tau"].cast<float_t>() : 1.0;
    auto applyNormOptional = agentArgs_.contains("apply_norm")
                             ? agentArgs_["apply_norm"].cast<std::string>()
                             : "none";
    auto applyNormToOptional = agentArgs_.contains("apply_norm_to")
                               ? agentArgs_["apply_norm_to"].cast<std::optional<std::vector<std::string>>>()
                               : std::optional<std::vector<std::string>>();
    auto epsForNorm = agentArgs_.contains("eps") ? agentArgs_["eps"].cast<float_t>() : 5e-8;
    auto pForNorm = agentArgs_.contains("p") ? agentArgs_["p"].cast<int32_t>() : 0;
    auto dimForNorm = agentArgs_.contains("dim") ? agentArgs_["dim"].cast<int32_t>() : 0;

    int32_t applyNorm = normModeCodes_[applyNormOptional];
    int32_t applyNormTo = applyNormToOptional.has_value() ? normApplyToCodes_[applyNormToOptional.value()] : -1;

    auto optimizerName = optimizerArgsRetrieved_["optimizer"].cast<std::string>();
    auto optimizerCode = optimizers_[optimizerName];
    auto models = get_dqn_model();
    switch (optimizerCode) {
        case 0: {
            return get_adam_optim_agent(models.at(0), models.at(1), gamma, epsilon, minEpsilon,
                                        epsilonDecayRate, epsilonDecayFrequency, memoryBufferSize,
                                        targetModelUpdateRate, policyModelUpdateRate, batchSize, numActions,
                                        savePath, (float_t) tau, applyNorm, applyNormTo,
                                        (float_t) epsForNorm, pForNorm, dimForNorm);
        }
        case 1: {

            return get_rmsprop_optim_agent(models.at(0), models.at(1), gamma, epsilon, minEpsilon,
                                           epsilonDecayRate, epsilonDecayFrequency, memoryBufferSize,
                                           targetModelUpdateRate, policyModelUpdateRate, batchSize, numActions,
                                           savePath, (float_t) tau, applyNorm, applyNormTo,
                                           (float_t) epsForNorm, pForNorm, dimForNorm);
        }
        default:
            throw std::runtime_error("Invalid Optimizer name passed!");
    }
}

torch::Tensor GetDqnAgent::pybind_array_to_torch_tensor(pybind11::array_t<float_t> &array, pybind11::tuple &shape) {
    std::vector<int64_t> shapeVector;
    for (auto &shape_: shape) {
        shapeVector.push_back(shape_.cast<int64_t>());
    }

    auto tensor = torch::zeros(shapeVector);
    memmove(tensor.data_ptr(), array.data(), tensor.nbytes());

    auto tensorOptions = torch::TensorOptions().dtype(dataType_).device(device_);
    tensor = tensor.to(tensorOptions);
    return tensor;
}

std::vector<std::shared_ptr<dqn::DqnBase>>
GetDqnAgent::get_relu_dcqn1d_model(int32_t sequenceLength, std::vector<int32_t> &channels,
                                   std::vector<int32_t> &kernelSizesConv,
                                   std::vector<int32_t> &strideSizesConv,
                                   std::vector<int32_t> &dilationSizesConv,
                                   std::optional<std::vector<int32_t>> &kernelSizesPool,
                                   std::optional<std::vector<int32_t>> &strideSizesPool,
                                   std::optional<std::vector<int32_t>> &dilationSizesPool,
                                   float_t dropout, int32_t numActions, bool usePadding) {
    // When activation is ReLU
    torch::nn::ReLU relu = torch::nn::ReLU();
    auto targetModel = std::make_shared<dqn::Dcqn1d<torch::nn::ReLU>>(sequenceLength, channels, kernelSizesConv,
                                                                      strideSizesConv, dilationSizesConv, numActions,
                                                                      kernelSizesPool, strideSizesPool,
                                                                      dilationSizesPool,
                                                                      relu, dropout, usePadding);
    auto policyModel = std::make_shared<dqn::Dcqn1d<torch::nn::ReLU>>(sequenceLength, channels, kernelSizesConv,
                                                                      strideSizesConv, dilationSizesConv, numActions,
                                                                      kernelSizesPool, strideSizesPool,
                                                                      dilationSizesPool,
                                                                      relu, dropout, usePadding);

    targetModel->to(dataType_);
    policyModel->to(dataType_);
    targetModel->to(device_);
    policyModel->to(device_);

    std::vector<std::shared_ptr<dqn::DqnBase>> dqnModels = {targetModel, policyModel};
    return dqnModels;
}

std::vector<std::shared_ptr<dqn::DqnBase>>
GetDqnAgent::get_relu_dlqn1d_model(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                                   float_t dropout) {
    // When activation is ReLU
    torch::nn::ReLU relu = torch::nn::ReLU();
    auto targetModel = std::make_shared<dqn::Dlqn1d<torch::nn::ReLU>>(sequenceLength,
                                                                      hiddenSizes, numAction,
                                                                      relu, dropout);

    auto policyModel = std::make_shared<dqn::Dlqn1d<torch::nn::ReLU>>(sequenceLength,
                                                                      hiddenSizes, numAction,
                                                                      relu, dropout);

    targetModel->to(dataType_);
    policyModel->to(dataType_);
    targetModel->to(device_);
    policyModel->to(device_);

    std::vector<std::shared_ptr<dqn::DqnBase>> dqnModels = {targetModel, policyModel};
    return dqnModels;
}

std::vector<std::shared_ptr<dqn::DqnBase>> GetDqnAgent::get_leaky_relu_dcqn1d_model(int32_t sequenceLength,
                                                                                    std::vector<int32_t> &channels,
                                                                                    std::vector<int32_t> &kernelSizesConv,
                                                                                    std::vector<int32_t> &strideSizesConv,
                                                                                    std::vector<int32_t> &dilationSizesConv,
                                                                                    std::optional<std::vector<int32_t>> &kernelSizesPool,
                                                                                    std::optional<std::vector<int32_t>> &strideSizesPool,
                                                                                    std::optional<std::vector<int32_t>> &dilationSizesPool,
                                                                                    float_t dropout,
                                                                                    int32_t numActions,
                                                                                    bool usePadding) {

    // When activation is LeakyReLU
    torch::nn::LeakyReLUOptions leakyReluOptions = torch::nn::LeakyReLUOptions();
    auto leakyReluArgs = activationArgs_["leaky_relu"];

    if (activationArgsRetrieved_.contains(leakyReluArgs.at(0))) {
        auto negativeSlope = activationArgsRetrieved_[leakyReluArgs.at(0).c_str()].cast<float_t>();
        leakyReluOptions.negative_slope(negativeSlope);
    }

    torch::nn::LeakyReLU leakyRelu = torch::nn::LeakyReLU(leakyReluOptions);
    auto targetModel = std::make_shared<dqn::Dcqn1d<torch::nn::LeakyReLU>>(sequenceLength, channels, kernelSizesConv,
                                                                           strideSizesConv, dilationSizesConv,
                                                                           numActions,
                                                                           kernelSizesPool, strideSizesPool,
                                                                           dilationSizesPool,
                                                                           leakyRelu, dropout, usePadding);

    auto policyModel = std::make_shared<dqn::Dcqn1d<torch::nn::LeakyReLU>>(sequenceLength, channels, kernelSizesConv,
                                                                           strideSizesConv, dilationSizesConv,
                                                                           numActions,
                                                                           kernelSizesPool, strideSizesPool,
                                                                           dilationSizesPool,
                                                                           leakyRelu, dropout, usePadding);

    targetModel->to(dataType_);
    policyModel->to(dataType_);
    targetModel->to(device_);
    policyModel->to(device_);

    std::vector<std::shared_ptr<dqn::DqnBase>> dqnModels = {targetModel, policyModel};
    return dqnModels;
}

std::vector<std::shared_ptr<dqn::DqnBase>>
GetDqnAgent::get_leaky_relu_dlqn1d_model(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                                         float_t dropout) {
    // When activation is Leaky ReLU
    torch::nn::LeakyReLU leakyRelu = torch::nn::LeakyReLU();
    auto targetModel = std::make_shared<dqn::Dlqn1d<torch::nn::LeakyReLU>>(sequenceLength,
                                                                           hiddenSizes, numAction,
                                                                           leakyRelu, dropout);

    auto policyModel = std::make_shared<dqn::Dlqn1d<torch::nn::LeakyReLU>>(sequenceLength,
                                                                           hiddenSizes, numAction,
                                                                           leakyRelu, dropout);

    targetModel->to(dataType_);
    policyModel->to(dataType_);
    targetModel->to(device_);
    policyModel->to(device_);

    std::vector<std::shared_ptr<dqn::DqnBase>> dqnModels = {targetModel, policyModel};
    return dqnModels;
}

std::vector<std::shared_ptr<dqn::DqnBase>>
GetDqnAgent::get_sigmoid_dcqn1d_model(int32_t sequenceLength, std::vector<int32_t> &channels,
                                      std::vector<int32_t> &kernelSizesConv,
                                      std::vector<int32_t> &strideSizesConv,
                                      std::vector<int32_t> &dilationSizesConv,
                                      std::optional<std::vector<int32_t>> &kernelSizesPool,
                                      std::optional<std::vector<int32_t>> &strideSizesPool,
                                      std::optional<std::vector<int32_t>> &dilationSizesPool,
                                      float_t dropout, int32_t numActions, bool usePadding) {
    // When activation is sigmoid.
    torch::nn::Sigmoid sigmoid = torch::nn::Sigmoid();
    auto targetModel = std::make_shared<dqn::Dcqn1d<torch::nn::Sigmoid>>(
            sequenceLength, channels, kernelSizesConv,
            strideSizesConv, dilationSizesConv,
            numActions, kernelSizesPool, strideSizesPool,
            dilationSizesPool, sigmoid, dropout, usePadding
    );
    auto policyModel = std::make_shared<dqn::Dcqn1d<torch::nn::Sigmoid>>(
            sequenceLength, channels, kernelSizesConv,
            strideSizesConv, dilationSizesConv,
            numActions, kernelSizesPool, strideSizesPool,
            dilationSizesPool, sigmoid, dropout, usePadding
    );
    targetModel->to(dataType_);
    policyModel->to(dataType_);
    targetModel->to(device_);
    policyModel->to(device_);

    std::vector<std::shared_ptr<dqn::DqnBase>> dqnModels = {targetModel, policyModel};
    return dqnModels;
}


std::vector<std::shared_ptr<dqn::DqnBase>>
GetDqnAgent::get_sigmoid_dlqn1d_model(int32_t sequenceLength, std::vector<int32_t> &hiddenSizes, int32_t numAction,
                                      float_t dropout) {
    // When activation is Leaky ReLU
    torch::nn::Sigmoid sigmoid = torch::nn::Sigmoid();
    auto targetModel = std::make_shared<dqn::Dlqn1d<torch::nn::Sigmoid>>(sequenceLength,
                                                                         hiddenSizes, numAction,
                                                                         sigmoid, dropout);

    auto policyModel = std::make_shared<dqn::Dlqn1d<torch::nn::Sigmoid>>(sequenceLength,
                                                                         hiddenSizes, numAction,
                                                                         sigmoid, dropout);

    targetModel->to(dataType_);
    policyModel->to(dataType_);
    targetModel->to(device_);
    policyModel->to(device_);

    std::vector<std::shared_ptr<dqn::DqnBase>> dqnModels = {targetModel, policyModel};
    return dqnModels;
}

AgentBase *GetDqnAgent::get_adam_optim_agent(std::shared_ptr<dqn::DqnBase> &targetModel,
                                             std::shared_ptr<dqn::DqnBase> &policyModel,
                                             float_t gamma, float_t epsilon, float_t minEpsilon,
                                             float_t epsilonDecayRate, int32_t epsilonDecayFrequency,
                                             int32_t memoryBufferSize, int32_t targetModelUpdateRate,
                                             int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
                                             std::string &savePath, float_t tau, int32_t applyNorm,
                                             int32_t applyNormTo, float_t epsForNorm, int32_t pForNorm,
                                             int32_t dimForNorm) {
    // When optimizer is Adam
    auto optimizersArgs = optimizersArgs_["adam"];
    auto lr = optimizerArgsRetrieved_[optimizersArgs.at(0).c_str()].cast<pybind11::float_>().cast<float_t>();
    torch::optim::AdamOptions adamOptions = torch::optim::AdamOptions(lr);

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(1).c_str())) {
        adamOptions.betas(
                optimizerArgsRetrieved_[optimizersArgs.at(1).c_str()].cast<std::tuple<float_t, float_t>>());
    }

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(2).c_str())) {
        adamOptions.eps(optimizerArgsRetrieved_[optimizersArgs.at(2).c_str()].cast<float_t>());
    }

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(3).c_str())) {
        adamOptions.weight_decay(optimizerArgsRetrieved_[optimizersArgs.at(3).c_str()].cast<float_t>());
    }

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(4).c_str())) {
        adamOptions.amsgrad(optimizerArgsRetrieved_[optimizersArgs.at(4).c_str()].cast<bool>());
    }

    auto adamOptimizer = std::make_shared<torch::optim::Adam>(
            policyModel->parameters(), adamOptions
    );

    auto *agent = new dqn::Agent<std::shared_ptr<dqn::DqnBase>, std::shared_ptr<torch::optim::Adam>>(
            targetModel, policyModel, adamOptimizer, gamma, epsilon, minEpsilon, epsilonDecayRate,
            epsilonDecayFrequency, memoryBufferSize, targetModelUpdateRate, policyModelUpdateRate,
            batchSize, numActions, savePath, tau, applyNorm, applyNormTo, epsForNorm, pForNorm, dimForNorm
    );
    return agent;
}

AgentBase *GetDqnAgent::get_rmsprop_optim_agent(std::shared_ptr<dqn::DqnBase> &targetModel,
                                                std::shared_ptr<dqn::DqnBase> &policyModel,
                                                float_t gamma, float_t epsilon, float_t minEpsilon,
                                                float_t epsilonDecayRate, int32_t epsilonDecayFrequency,
                                                int32_t memoryBufferSize, int32_t targetModelUpdateRate,
                                                int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
                                                std::string &savePath, float_t tau, int32_t applyNorm,
                                                int32_t applyNormTo, float_t epsForNorm, int32_t pForNorm,
                                                int32_t dimForNorm) {

    // When optimizer is RMSProp
    auto optimizersArgs = optimizersArgs_["rmsprop"];
    auto lr = optimizerArgsRetrieved_[optimizersArgs.at(0).c_str()].cast<float_t>();
    torch::optim::RMSpropOptions rmsPropOptions = torch::optim::RMSpropOptions(lr);

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(1).c_str())) {
        rmsPropOptions.alpha(optimizerArgsRetrieved_[optimizersArgs.at(1).c_str()].cast<float_t>());
    }

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(2).c_str())) {
        rmsPropOptions.eps(optimizerArgsRetrieved_[optimizersArgs.at(2).c_str()].cast<float_t>());
    }

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(3).c_str())) {
        rmsPropOptions.weight_decay(optimizerArgsRetrieved_[optimizersArgs.at(3).c_str()].cast<float_t>());
    }

    if (optimizerArgsRetrieved_.contains(optimizersArgs.at(4).c_str())) {
        rmsPropOptions.momentum(optimizerArgsRetrieved_[optimizersArgs.at(4).c_str()].cast<float_t>());
    }

    auto rmsPropOptimizer = std::make_shared<torch::optim::RMSprop>(
            policyModel->parameters(), rmsPropOptions
    );

    auto *agent = new dqn::Agent<std::shared_ptr<dqn::DqnBase>, std::shared_ptr<torch::optim::RMSprop>>(
            targetModel, policyModel, rmsPropOptimizer, gamma, epsilon, minEpsilon, epsilonDecayRate,
            epsilonDecayFrequency, memoryBufferSize, targetModelUpdateRate, policyModelUpdateRate,
            batchSize, numActions, savePath, tau, applyNorm, applyNormTo, epsForNorm, pForNorm, dimForNorm
    );

    return agent;
}


torch::DeviceType GetDqnAgent::get_device(std::string &device) {
    if (device == "mps") {
        return torch::kMPS;
    }
    if (device == "cuda") {
        return torch::kCUDA;
    }

    return torch::kCPU;
}

torch::ScalarType GetDqnAgent::get_data_type(std::string &device) {
    if (device == "mps") {
        return torch::kFloat32;
    }
    if (device == "cuda") {
        return torch::kDouble;
    }

    return torch::kDouble;
}

GetDqnAgent::~GetDqnAgent() = default;

#endif//RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_