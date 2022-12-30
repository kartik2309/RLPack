//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_

#include <torch/extension.h>
#include "../utils/maps.h"

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
  * @brief The class C_RolloutBuffer is the class that implements the C++ backend for Rollout Buffer. Tensors are moved
  * to C++ backend via PyBind11 and are kept opaque with std::map, hence, tensors are moved between Python and C++ only
  * by references. `C_RolloutBuffer` is hence autograd safe.
  */
class C_RolloutBuffer {

public:
    C_RolloutBuffer(int64_t bufferSize, std::string &device, std::string &dtype);
    ~C_RolloutBuffer();

    void insert(std::map<std::string, torch::Tensor> &inputMap);
    std::map<std::string, torch::Tensor> compute_returns(float_t gamma);
    std::map<std::string, torch::Tensor> get_stacked_rewards();
    std::map<std::string, torch::Tensor> get_stacked_action_log_probabilities();
    std::map<std::string, torch::Tensor> get_stacked_state_current_values();
    std::map<std::string, torch::Tensor> get_stacked_entropies();
    void clear();
    size_t size();

private:
    //! The buffer size that is going to be used.
    int64_t bufferSize_;
    //! The device that is to be used for PyTorch tensors.
    torch::DeviceType device_;
    //! The datatype to be used for PyTorch tensors.
    torch::Dtype dtype_;
    //! The tensor options to be used for PyTorch tensors; constructed with device_ and dtype_.
    torch::TensorOptions tensorOptions_;
    //! The vector of accumulated rewards.
    std::vector<torch::Tensor> rewards_;
    //! The vector of action log probabilities.
    std::vector<torch::Tensor> actionLogProbabilities_;
    //! The vector of current state values.
    std::vector<torch::Tensor> stateCurrentValues_;
    //! The vector of entropies of current distribution.
    std::vector<torch::Tensor> entropies_;
};
/*!
 * @} @I{ // End group rollout_buffer_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_
