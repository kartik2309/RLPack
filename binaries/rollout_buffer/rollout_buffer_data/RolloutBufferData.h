//
// Created by Kartik Rajeshwaran on 2023-01-03.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_DATA_ROLLOUTBUFFERDATA_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_DATA_ROLLOUTBUFFERDATA_H_

#define STATES_CURRENT "states_current"
#define STATE_CURRENT "state_current"
#define STATES_NEXT "states_next"
#define STATE_NEXT "state_next"
#define REWARDS "rewards"
#define REWARD "reward"
#define DONES "dones"
#define DONE "done"
#define ACTION_LOG_PROBABILITIES "action_log_probabilities"
#define ACTION_LOG_PROBABILITY "action_log_probability"
#define STATE_CURRENT_VALUES "state_current_values"
#define STATE_CURRENT_VALUE "state_current_value"
#define STATE_NEXT_VALUES "state_next_values"
#define STATE_NEXT_VALUE "state_next_value"
#define ENTROPIES "entropies"
#define ENTROPY "entropy"
#define RETURNS "returns"
#define TD_RESIDUALS "td_residuals"
#define ADVANTAGES "advantages"

#include <torch/extension.h>

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
  * @brief The class RolloutBufferData keeps the reference of each tensor of RolloutBufferContainer and provides necessary
  * methods to ship tensors to Python
  */
class RolloutBufferData {
public:
    //! The pointer to current state tensor
    torch::Tensor* stateCurrent = nullptr;
    //! The pointer to next state.
    torch::Tensor* stateNext = nullptr;
    //! The pointer to reward value
    torch::Tensor* reward = nullptr;
    //! The pointer to done value
    torch::Tensor* done = nullptr;
    //! The pointer to action log probability.
    torch::Tensor* actionLogProbability = nullptr;
    //! The pointer to current state value.
    torch::Tensor* stateCurrentValue = nullptr;
    //! The pointer to next state value.
    torch::Tensor* stateNextValue = nullptr;
    //! The pointer to entropy of current distribution.
    torch::Tensor* entropy = nullptr;

    RolloutBufferData();
    ~RolloutBufferData();

    [[nodiscard]] std::map<std::string, torch::Tensor> get_transition_data(torch::TensorOptions& tensorOptions) const;
    [[nodiscard]] std::map<std::string, torch::Tensor> get_policy_output_data() const;
};
/*!
 * @} @I{ // End group rollout_buffer_group }
 * @} @I{ // End group binaries_group }
 */

#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_DATA_ROLLOUTBUFFERDATA_H_
