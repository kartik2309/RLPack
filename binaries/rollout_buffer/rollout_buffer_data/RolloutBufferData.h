//
// Created by Kartik Rajeshwaran on 2023-01-03.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_DATA_ROLLOUTBUFFERDATA_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_DATA_ROLLOUTBUFFERDATA_H_

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

//! Macro for states current for easy use in tensor map.
#define STATES_CURRENT "states_current"
//! Macro for state current for easy use in tensor map.
#define STATE_CURRENT "state_current"
//! Macro for states next for easy use in tensor map.
#define STATES_NEXT "states_next"
//! Macro for state next for easy use in tensor map.
#define STATE_NEXT "state_next"
//! Macro for rewards for easy use in tensor map.
#define REWARDS "rewards"
//! Macro for reward for easy use in tensor map.
#define REWARD "reward"
//! Macro for dones for easy use in tensor map.
#define DONES "dones"
//! Macro for done for easy use in tensor map.
#define DONE "done"
//! Macro for action log probabilities for easy use in tensor map.
#define ACTION_LOG_PROBABILITIES "action_log_probabilities"
//! Macro for action log probability for easy use in tensor map.
#define ACTION_LOG_PROBABILITY "action_log_probability"
//! Macro for state current values for easy use in tensor map.
#define STATE_CURRENT_VALUES "state_current_values"
//! Macro for state current value for easy use in tensor map.
#define STATE_CURRENT_VALUE "state_current_value"
//! Macro for state next values for easy use in tensor map.
#define STATE_NEXT_VALUES "state_next_values"
//! Macro for state next value for easy use in tensor map.
#define STATE_NEXT_VALUE "state_next_value"
//! Macro for entropies for easy use in tensor map.
#define ENTROPIES "entropies"
//! Macro for entropy for easy use in tensor map.
#define ENTROPY "entropy"
//! Macro for returns for easy use in tensor map.
#define RETURNS "returns"
//! Macro for TD Rrsiduals for easy use in tensor map.
#define TD_RESIDUALS "td_residuals"
//! Macro for advantages for easy use in tensor map.
#define ADVANTAGES "advantages"

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
