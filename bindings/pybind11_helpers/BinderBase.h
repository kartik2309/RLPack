//
// Created by Kartik Rajeshwaran on 2022-07-08.
//


#ifndef RLPACK_BINDINGS_PYBIND11_HELPERS_RLPACK_BINDERBASE_H
#define RLPACK_BINDINGS_PYBIND11_HELPERS_RLPACK_BINDERBASE_H

#include <map>
#include <vector>
#include <string>
#include <pybind11/numpy.h>
# include "../../src/Optimizers/Optimizer.hpp"
# include "../../src/AgentBase.h"
# include "../../src/ModelBase.h"
# include "../../src/utils/Options/AgentOptions/AgentOptionsBase.h"
# include "../../src/utils/Options/ModelOptions/ModelOptionsBase.h"

class BinderBase {

protected:
    std::map<std::string, int> activations_ = {
            {"relu",       0},
            {"leaky_relu", 1},
            {"sigmoid",    2}
    };
    std::map<std::string, std::vector<std::string>> activationArgs_ = {
            {"relu",       {}},
            {"leaky_relu", {"negative_slope"}}
    };

    std::map<std::string, int> optimizers_ = {
            {"adam",    0},
            {"rmsprop", 1},
            {"Sgd",     2}
    };

    std::map<std::string, int> lrSchedulers_ = {
            {"step_lr", 0},
    };

    std::map<std::string, std::vector<std::string>> optimizersArgs_ = {
            {"adam",    {"lr", "betas",    "eps",       "weight_decay", "amsgrad"}},
            {"rmsprop", {"lr", "alpha",    "eps",       "weight_decay", "momentum"}},
            {"sgd",     {"lr", "momentum", "dampening", "weight_decay", "nesterov"}}
    };

    std::map<std::string, std::vector<std::string>> lrSchedulerArgs_ = {
            {"step_lr", {"optim", "step_size", "gamma", "last_epoch"}}
    };

    std::map<std::string, int> normModeCodes_ = {
            {"none",        -1},
            {"min_max",     0},
            {"standardize", 1},
            {"p_norm",      2}
    };

    std::map<std::vector<std::string>, int> normApplyToCodes_ = {
            {{"none"},              -1},
            {{"states"},            0},
            {{"rewards"},           1},
            {{"td"},                2},
            {{"states", "rewards"}, 3},
            {{"states", "td"},      4}
    };

    std::map<std::string, int> deviceCodes_ = {
            {"cpu",  0},
            {"cuda", 1},
            {"mps",  2},
    };

    std::map<int, torch::DeviceType> deviceTypes_ = {
            {0, torch::kCPU},
            {1, torch::kCUDA},
            {2, torch::kMPS},
    };

    std::map<int, torch::ScalarType> deviceDataTypes_{
            {0, torch::kFloat64},
            {1, torch::kFloat64},
            {2, torch::kFloat32},
    };

    static torch::Tensor pybind_array_to_torch_tensor(
            pybind11::array_t<float_t> &array, pybind11::tuple &shape,
            torch::ScalarType dataType, torch::DeviceType device
    );

    torch::DeviceType get_device(std::string &device);

    torch::ScalarType get_data_type(std::string &device);

    virtual std::unique_ptr<agent::AgentOptionsBase> create_agent_options() = 0;

    virtual std::unique_ptr<model::ModelOptionsBase> create_model_options() = 0;

    virtual std::shared_ptr<optimizer::OptimizerBase> create_optimizer(std::shared_ptr<model::ModelBase> &model) = 0;

    virtual std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase>
    create_lr_scheduler(std::shared_ptr<optimizer::OptimizerBase> &optimizer) = 0;

    virtual std::optional<std::shared_ptr<ActivationBase>> create_activation_module() = 0;

    virtual std::shared_ptr<model::ModelBase> create_model() = 0;

    virtual std::shared_ptr<agent::AgentBase> create_agent() = 0;

};

#endif //RLPACK_BINDINGS_PYBIND11_HELPERS_RLPACK_BINDERBASE_H
