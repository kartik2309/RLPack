//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_

#include "rollout_buffer_batch/RolloutBufferBatch.h"
#include "rollout_buffer_batch_transform/RolloutBufferBatchTransform.h"
#include "rollout_buffer_container/RolloutBufferContainer.h"

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
    //! The Torch DataLoader unique pointer type for RolloutBuffer.
    using DataLoader = torch::disable_if_t<
            false,
            std::unique_ptr<torch::data::StatelessDataLoader<
                    torch::data::datasets::MapDataset<RolloutBufferBatch, RolloutBufferBatchTransform>,
                    torch::data::samplers::SequentialSampler>>>;

    C_RolloutBuffer(int64_t bufferSize,
                    std::string& device,
                    std::string& dtype);
    ~C_RolloutBuffer();

    void insert_transition(std::map<std::string, torch::Tensor>& inputMap);
    void insert_policy_output(std::map<std::string, torch::Tensor>& inputMap);
    std::map<std::string, torch::Tensor> compute_returns(float_t gamma);
    std::map<std::string, torch::Tensor> compute_discounted_td_residuals(float_t gamma);
    std::map<std::string, torch::Tensor> compute_generalized_advantage_estimates(float_t gamma, float_t gaeLambda);
    std::map<std::string, torch::Tensor> get_stacked_states_current();
    std::map<std::string, torch::Tensor> get_stacked_states_next();
    std::map<std::string, torch::Tensor> get_stacked_rewards();
    std::map<std::string, torch::Tensor> get_stacked_dones();
    std::map<std::string, torch::Tensor> get_stacked_action_log_probabilities();
    std::map<std::string, torch::Tensor> get_stacked_state_current_values();
    std::map<std::string, torch::Tensor> get_stacked_state_next_values();
    std::map<std::string, torch::Tensor> get_stacked_entropies();
    std::map<std::string, torch::Tensor> get_states_statistics();
    std::map<std::string, torch::Tensor> get_advantage_statistics(float_t gamma, float_t gae_lambda);
    std::map<std::string, torch::Tensor> get_action_log_probabilities_statistics();
    std::map<std::string, torch::Tensor> get_state_values_statistics();
    std::map<std::string, torch::Tensor> get_entropy_statistics();
    std::map<std::string, torch::Tensor> transition_at(int64_t index);
    std::map<std::string, torch::Tensor> policy_output_at(int64_t index);
    void clear_transitions();
    void clear_policy_outputs();
    size_t size_transitions();
    size_t size_policy_outputs();
    void extend_transitions(std::map<std::string, std::vector<torch::Tensor>>& extensionMap);
    void set_transitions_iterator(int64_t batchSize);
    DataLoader& get_dataloader_reference();

private:
    //! The pointer to dynamically allocated RolloutBufferContainer object.
    RolloutBufferContainer* rolloutBufferContainer_;
    //! The tensor options to be used for PyTorch tensors; constructed with device_ and dtype_.
    torch::TensorOptions tensorOptions_;
    //! The DataLoader object. This is initialized to nullptr until `set_transitions_iterator` is called.
    DataLoader dataloader_ = nullptr;
};
/*!
 * @} @I{ // End group rollout_buffer_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_
