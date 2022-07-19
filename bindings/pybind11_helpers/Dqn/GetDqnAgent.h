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

#include "../../../src/utils/Activations/Activations.hpp"
#include "../../../src/Optimizers/Optimizer.hpp"
#include "../../../src/utils/Base/AgentBase/AgentBase.h"
#include "../../../src/Dqn/Dqn.hpp"
#include "../BinderBase.h"

class GetDqnAgent : protected BinderBase {
public:
    GetDqnAgent(
            pybind11::str &modelName, pybind11::dict &modelArgs, pybind11::dict &activationArgs,
            pybind11::dict &agentArgs, pybind11::dict &optimizerArgs, pybind11::dict &lrSchedulerArgs,
            pybind11::str &device
    );

    ~GetDqnAgent();

    int train(pybind11::array_t<float_t> &stateCurrent, pybind11::array_t<float_t> &stateNext,
              pybind11::float_ &reward, pybind11::int_ &action, pybind11::bool_ &done,
              pybind11::tuple &stateCurrentShape, pybind11::tuple &stateNextShape);

    int policy(pybind11::array_t<float_t> &stateCurrent, pybind11::tuple &stateCurrentShape);

    void setup_agent();

    void save();

    void load();

private:
    std::string modelName_;
    pybind11::kwargs modelArgs_;
    pybind11::kwargs activationArgsRetrieved_;
    pybind11::kwargs agentArgs_;
    pybind11::kwargs optimizerArgsRetrieved_;
    pybind11::kwargs lrSchedulerArgsRetrieved_;
    std::shared_ptr<agent::AgentBase> agent_;
    torch::DeviceType device_;
    torch::ScalarType dataType_;

    std::map<std::string, int> dqnModels_ = {{"dcqn1d", 0},
                                             {"dlqn1d", 1}};


    std::unique_ptr<agent::AgentOptionsBase> create_agent_options() override;

    std::unique_ptr<model::ModelOptionsBase> create_model_options() override;

    std::shared_ptr<optimizer::OptimizerBase> create_optimizer(std::shared_ptr<model::ModelBase> &model) override;

    std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase>
    create_lr_scheduler(std::shared_ptr<optimizer::OptimizerBase> &optimizer) override;

    std::optional<std::shared_ptr<ActivationBase>> create_activation_module() override;

    std::shared_ptr<model::ModelBase> create_model() override;

    std::shared_ptr<agent::AgentBase> create_agent() override;

};

#endif//RLPACK_BINDINGS_PYBIND11_HELPERS_DQN_GETDQNAGENT_H_
