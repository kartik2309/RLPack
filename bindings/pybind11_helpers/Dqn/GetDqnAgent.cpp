//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

//#ifndef RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_
//#define RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_

#include "GetDqnAgent.h"


GetDqnAgent::GetDqnAgent(
        pybind11::str &modelName, pybind11::dict &modelArgs, pybind11::dict &activationArgs,
        pybind11::dict &agentArgs, pybind11::dict &optimizerArgs, pybind11::dict &lrSchedulerArgs,
        pybind11::str &device
) {
    modelName_ = modelName.cast<std::string>();
    modelArgs_ = modelArgs;
    activationArgsRetrieved_ = activationArgs;
    agentArgs_ = agentArgs;
    optimizerArgsRetrieved_ = optimizerArgs;
    lrSchedulerArgsRetrieved_ = lrSchedulerArgs;

    auto deviceName = device.cast<std::string>();
    device_ = get_device(deviceName);
    dataType_ = get_data_type(deviceName);
}

int GetDqnAgent::train(pybind11::array_t<float_t> &stateCurrent, pybind11::array_t<float_t> &stateNext,
                       pybind11::float_ &reward, pybind11::int_ &action, pybind11::bool_ &done,
                       pybind11::tuple &stateCurrentShape, pybind11::tuple &stateNextShape) {

    // Convert pybind::array (numpy arrays) to torch::Tensor.
    auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape, dataType_, device_);
    auto stateNextTensor_ = pybind_array_to_torch_tensor(stateNext, stateNextShape, dataType_, device_);

    // Convert other arguments to native C++ types.
    auto reward_ = reward.cast<float>();
    auto action_ = action.cast<int>();
    auto done_ = done.cast<int>();

    action_ = agent_->train(stateCurrentTensor_, stateNextTensor_, reward_, action_, done_);
    return action_;
}

int GetDqnAgent::policy(pybind11::array_t<float_t> &stateCurrent, pybind11::tuple &stateCurrentShape) {
    auto stateCurrentTensor_ = pybind_array_to_torch_tensor(stateCurrent, stateCurrentShape, dataType_, device_);
    auto action = agent_->policy(stateCurrentTensor_);
    return action;
}

void GetDqnAgent::setup_agent() {
    agent_ = create_agent();
}

void GetDqnAgent::save() {
    agent_->save();
}

void GetDqnAgent::load() {
    agent_->load();
}

void GetDqnAgent::finish() {
    agent_->finish();
}

void GetDqnAgent::barrier() {
    agent_->barrier();
}

void GetDqnAgent::sync_models() {
    agent_->sync_models();
}


std::unique_ptr<agent::AgentOptionsBase> GetDqnAgent::create_agent_options() {
    auto gamma = agentArgs_["gamma"].cast<float_t>();
    auto epsilon = agentArgs_["epsilon"].cast<float_t>();
    auto minEpsilon = agentArgs_["min_epsilon"].cast<float_t>();
    auto epsilonDecayRate = agentArgs_["epsilon_decay_rate"].cast<float_t>();
    auto epsilonDecayFrequency = agentArgs_["epsilon_decay_frequency"].cast<int32_t>();
    auto memoryBufferSize = agentArgs_["memory_buffer_size"].cast<int32_t>();
    auto targetModelUpdateRate = agentArgs_["target_model_update_rate"].cast<int32_t>();
    auto policyModelUpdateRate = agentArgs_["policy_model_update_rate"].cast<int32_t>();
    auto modelBackupFrequency = agentArgs_["model_backup_frequency"].cast<int32_t>();
    auto minLr = agentArgs_["min_lr"].cast<float_t>();
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
    auto targetModel = create_model();
    auto policyModel = create_model();

    targetModel->to(device_);
    targetModel->to(dataType_);
    policyModel->to(device_);
    policyModel->to(dataType_);

    auto optimizer = create_optimizer(policyModel);
    auto lrScheduler = create_lr_scheduler(optimizer);

    auto agentOptions = std::make_unique<agent::dqn::DqnAgentOptions>();
    agentOptions->set_target_model(targetModel);
    agentOptions->set_policy_model(policyModel);
    agentOptions->set_optimizer(optimizer);
    agentOptions->set_lr_scheduler(lrScheduler);
    agentOptions->gamma(gamma);
    agentOptions->epsilon(epsilon);
    agentOptions->min_epsilon(minEpsilon);
    agentOptions->epsilon_decay_rate(epsilonDecayRate);
    agentOptions->epsilon_decay_frequency(epsilonDecayFrequency);
    agentOptions->memory_buffer_size(memoryBufferSize);
    agentOptions->target_model_update_rate(targetModelUpdateRate);
    agentOptions->policy_model_update_rate(policyModelUpdateRate);
    agentOptions->model_backup_frequency(modelBackupFrequency);
    agentOptions->min_lr(minLr);
    agentOptions->batch_size(batchSize);
    agentOptions->num_actions(numActions);
    agentOptions->d_type(dataType_);
    agentOptions->save_path(savePath);
    agentOptions->tau((float_t) tau);
    agentOptions->apply_norm(applyNorm);
    agentOptions->apply_norm_to(applyNormTo);
    agentOptions->eps_for_norm((float_t) epsForNorm);
    agentOptions->p_for_norm(pForNorm);
    agentOptions->dim_for_norm(dimForNorm);

    return agentOptions;
}

std::unique_ptr<model::ModelOptionsBase> GetDqnAgent::create_model_options() {

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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Common Arguments for Dcqn1d and Dlqn1d Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto sequenceLength = modelArgs_.contains("sequence_length")
                          ? modelArgs_["sequence_length"].cast<std::optional<int32_t>>()
                          : std::optional<int32_t>();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Common Arguments for all model::dqn Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto dropout = modelArgs_["dropout"].cast<float_t>();
    auto numActions = modelArgs_["num_actions"].cast<int32_t>();

    auto modelCode = dqnModels_[modelName_];

    switch (modelCode) {
        case 0: {
            auto modelOptions = std::make_unique<model::dqn::Dcqn1dOptions>();
            modelOptions->sequence_length(sequenceLength.value());
            modelOptions->channels(channels.value());
            modelOptions->conv_kernel_sizes(kernelSizesConv.value());
            modelOptions->conv_stride_sizes(strideSizesConv.value());
            modelOptions->conv_dilation_sizes(dilationSizesPool.value());
            modelOptions->pool_kernel_sizes(kernelSizesConv);
            modelOptions->pool_stride_sizes(strideSizesPool);
            modelOptions->pool_dilation_sizes(dilationSizesPool);
            modelOptions->use_padding(usePadding.value());
            modelOptions->num_actions(numActions);
            modelOptions->dropout(dropout);
            auto activationModule = create_activation_module();
            modelOptions->activation(activationModule);

            return modelOptions;
        }
        case 1: {
            auto modelOptions = std::make_unique<model::dqn::Dlqn1dOptions>();
            modelOptions->sequence_length(sequenceLength.value());
            modelOptions->hidden_sizes(hiddenSizes.value());
            modelOptions->num_actions(numActions);
            modelOptions->dropout(dropout);
            auto activationModule = create_activation_module();
            modelOptions->activation(activationModule);

            return modelOptions;
        }
        default:
            throw std::runtime_error("Invalid or unsupported model_name was passed!");
    }
}

std::optional<std::shared_ptr<ActivationBase>> GetDqnAgent::create_activation_module() {
    auto activation = modelArgs_["activation"].cast<std::string>();
    auto activationCode = activations_[activation];
    switch (activationCode) {
        case 0: {
            auto activationModule = std::make_shared<Relu>();
            return activationModule;
        }
        case 1: {
            torch::nn::LeakyReLUOptions leakyReLuOptions = torch::nn::LeakyReLUOptions();
            if (activationArgsRetrieved_.contains(activationArgs_[activation].at(0))) {
                auto negativeSlope = activationArgsRetrieved_[
                        activationArgs_[activation].at(0).c_str()].cast<float_t>();
                leakyReLuOptions.negative_slope(negativeSlope);
            }
            auto activationModule = std::make_shared<LeakyRelu>(leakyReLuOptions);
            return activationModule;
        }
        default:
            BOOST_LOG_TRIVIAL(warning) << "Invalid or unsupported activation was passed!"
                                          " Will default to ReLU" << std::endl;
            return {};
    }
}

std::shared_ptr<model::ModelBase> GetDqnAgent::create_model() {
    auto modelCode = dqnModels_[modelName_];
    switch (modelCode) {
        case 0: {
            auto dcqn1dOptionsBase = create_model_options();
            std::unique_ptr<model::dqn::Dcqn1dOptions> dcqn1dOptions(
                    (model::dqn::Dcqn1dOptions *) dcqn1dOptionsBase.release());
            auto dcqnModel = std::make_shared<model::dqn::Dcqn1d>(dcqn1dOptions);
            return dcqnModel;
        }
        case 1: {
            auto dlqn1dOptionsBase = create_model_options();
            std::unique_ptr<model::dqn::Dlqn1dOptions> dlqn1dOptions(
                    (model::dqn::Dlqn1dOptions *) dlqn1dOptionsBase.release());
            auto dlqnModel = std::make_shared<model::dqn::Dlqn1d>(dlqn1dOptions);
            return dlqnModel;
        }
        default:
            throw std::runtime_error("Invalid or unsupported model was passed");
    }
}

std::shared_ptr<agent::AgentBase> GetDqnAgent::create_agent() {
    auto agentOptionsBase = create_agent_options();
    std::unique_ptr<agent::dqn::DqnAgentOptions> agentOptions(
            (agent::dqn::DqnAgentOptions *) agentOptionsBase.release());
    auto agent = std::make_shared<agent::dqn::Agent>(agentOptions);
    return agent;
}

std::shared_ptr<optimizer::OptimizerBase> GetDqnAgent::create_optimizer(std::shared_ptr<model::ModelBase> &model) {
    auto optimizerName = optimizerArgsRetrieved_["optimizer"].cast<std::string>();
    auto optimizerCode = optimizers_[optimizerName];

    switch (optimizerCode) {
        case 0: {

            auto optimizersArgs = optimizersArgs_["adam"];
            auto lr = optimizerArgsRetrieved_[optimizersArgs.at(0).c_str()].cast<pybind11::float_>().cast<float_t>();
            auto adamOptions = std::make_shared<torch::optim::AdamOptions>(lr);

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(1).c_str())) {
                adamOptions->betas(
                        optimizerArgsRetrieved_[optimizersArgs.at(1).c_str()].cast<std::tuple<float_t, float_t>>());
            }

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(2).c_str())) {
                adamOptions->eps(optimizerArgsRetrieved_[optimizersArgs.at(2).c_str()].cast<float_t>());
            }

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(3).c_str())) {
                adamOptions->weight_decay(optimizerArgsRetrieved_[optimizersArgs.at(3).c_str()].cast<float_t>());
            }

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(4).c_str())) {
                adamOptions->amsgrad(optimizerArgsRetrieved_[optimizersArgs.at(4).c_str()].cast<bool>());
            }

            auto adamOptimizer = std::make_shared<optimizer::Adam>(
                    model->parameters(), adamOptions
            );
            return adamOptimizer;
        }
        case 1: {
            auto optimizersArgs = optimizersArgs_["rmsprop"];
            auto lr = optimizerArgsRetrieved_[optimizersArgs.at(0).c_str()].cast<float_t>();
            auto rmsPropOptions = std::make_shared<torch::optim::RMSpropOptions>(lr);

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(1).c_str())) {
                rmsPropOptions->alpha(optimizerArgsRetrieved_[optimizersArgs.at(1).c_str()].cast<float_t>());
            }

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(2).c_str())) {
                rmsPropOptions->eps(optimizerArgsRetrieved_[optimizersArgs.at(2).c_str()].cast<float_t>());
            }

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(3).c_str())) {
                rmsPropOptions->weight_decay(optimizerArgsRetrieved_[optimizersArgs.at(3).c_str()].cast<float_t>());
            }

            if (optimizerArgsRetrieved_.contains(optimizersArgs.at(4).c_str())) {
                rmsPropOptions->momentum(optimizerArgsRetrieved_[optimizersArgs.at(4).c_str()].cast<float_t>());
            }

            auto rmsPropOptimizer = std::make_shared<optimizer::RmsProp>(
                    model->parameters(), rmsPropOptions
            );

            return rmsPropOptimizer;
        }
        default:
            throw std::runtime_error("Invalid or unsupported optim passed!");
    }

}

std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase>
GetDqnAgent::create_lr_scheduler(std::shared_ptr<optimizer::OptimizerBase> &optimizer) {
    int lrSchedulerCode;
    if (lrSchedulerArgsRetrieved_.contains("scheduler")) {
        auto lrSchedulerName = lrSchedulerArgsRetrieved_["scheduler"].cast<std::string>();
        lrSchedulerCode = lrSchedulers_[lrSchedulerName];
    } else {
        return nullptr;
    }

    switch (lrSchedulerCode) {
        case 0: {
            auto stepLrSchedulerOptions = std::make_shared<optimizer::lrScheduler::StepLrOptions>();

            stepLrSchedulerOptions->optimizer(optimizer);

            if (lrSchedulerArgsRetrieved_.contains("step_size")) {
                auto stepSize = lrSchedulerArgsRetrieved_["step_size"].cast<uint32_t>();
                stepLrSchedulerOptions->step_size(stepSize);
            }

            if (lrSchedulerArgsRetrieved_.contains("gamma")) {
                auto gamma = lrSchedulerArgsRetrieved_["gamma"].cast<const float_t>();
                stepLrSchedulerOptions->gamma(gamma);
            }

            auto stepLrScheduler = std::make_shared<optimizer::lrScheduler::StepLr>(stepLrSchedulerOptions);
            return stepLrScheduler;
        }
        default:
            return nullptr;
    }
}

GetDqnAgent::~GetDqnAgent() = default;

//#endif//RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_TPP_