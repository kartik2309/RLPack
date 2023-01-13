//
// Created by Kartik Rajeshwaran on 2023-01-03.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERCONTAINER_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERCONTAINER_H_

#include <torch/extension.h>

#include "../../utils/maps.h"
#include "../rollout_buffer_data/RolloutBufferData.h"

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup rollout_buffer_group rollout_buffer
 * @brief Rollout Buffer is the C++ backend for the class rlpack._C.rollout_buffer.RolloutBuffer. This module implements
 * necessary classes to provide necessary functionalities and bindings to provide exposure to Python.
 * @{
 */
/*!
  * @brief The class RolloutBufferContainer is the main backend container that is used in C_RolloutBuffer. This container stores
  * the data and tracks the references with RolloutBufferData objects to keep transitions and policy outputs seperated. It
  * also provides necessary methods to manipulate the buffer and get its properties.
  */
class RolloutBufferContainer {

public:

    RolloutBufferContainer(int64_t bufferSize, std::string &device, std::string &dtype);
    ~RolloutBufferContainer();

    void insert_transition(torch::Tensor &stateCurrent,
                           torch::Tensor &stateNext,
                           torch::Tensor &reward,
                           torch::Tensor &done);
    void insert_policy_output(torch::Tensor &actionLogProbability,
                              torch::Tensor &stateCurrentValue,
                              torch::Tensor &stateNextValue,
                              torch::Tensor &entropy);
    void clear_transitions();
    void clear_policy_outputs();
    RolloutBufferData transition_at(int64_t index);
    RolloutBufferData policy_output_at(int64_t index);
    std::vector<torch::Tensor> &get_states_current_reference();
    std::vector<torch::Tensor> &get_states_next_reference();
    std::vector<torch::Tensor> &get_rewards_reference();
    std::vector<torch::Tensor> &get_dones_reference();
    std::vector<torch::Tensor> &get_action_log_probabilities_reference();
    std::vector<torch::Tensor> &get_state_current_values_reference();
    std::vector<torch::Tensor> &get_state_next_values_reference();
    std::vector<torch::Tensor> &get_entropies_reference();
    torch::TensorOptions get_tensor_options();
    [[nodiscard]] int64_t get_buffer_size() const;
    size_t size_transitions();
    size_t size_policy_outputs();
    void extend_transitions(std::map<std::string, std::vector<torch::Tensor>> &extensionMap);

private:
    //! The buffer size that is going to be used.
    int64_t bufferSize_;
    //! The device that is to be used for PyTorch tensors.
    torch::DeviceType device_;
    //! The datatype to be used for PyTorch tensors.
    torch::Dtype dtype_;
    //! The tensor options to be used for PyTorch tensors; constructed with device_ and dtype_.
    torch::TensorOptions tensorOptions_;
    //! The vector of current states.
    std::vector<torch::Tensor> statesCurrent_;
    //! The vector of next states.
    std::vector<torch::Tensor> statesNext_;
    //! The vector of accumulated rewards.
    std::vector<torch::Tensor> rewards_;
    //! The vector of dones.
    std::vector<torch::Tensor> dones_;
    //! The vector of action log probabilities.
    std::vector<torch::Tensor> actionLogProbabilities_;
    //! The vector of current state values.
    std::vector<torch::Tensor> stateCurrentValues_;
    //! The vector of next state values.
    std::vector<torch::Tensor> stateNextValues_;
    //! The vector of entropies of current distribution.
    std::vector<torch::Tensor> entropies_;
    //! The vector of RolloutBufferData pointers for transition data
    std::vector<RolloutBufferData> transitionData_;
    //! The vector of RolloutBufferData pointers for policy outputs
    std::vector<RolloutBufferData> policyOutputData_;
};
/*!
 * @} @I{ // End group rollout_buffer_group }
 * @} @I{ // End group binaries_group }
 */

#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERCONTAINER_H_
