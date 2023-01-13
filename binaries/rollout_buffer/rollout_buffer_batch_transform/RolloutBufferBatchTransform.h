//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERBATCHTRANSFORM_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERBATCHTRANSFORM_H_

#include <torch/extension.h>

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
  * @brief The class RolloutBufferBatchTransform applies transform to each batch when using iterator to iterate through
  * each transition batch. This class inherits from `Collate` class in PyTorch.
  */
class RolloutBufferBatchTransform
    : public torch::data::transforms::Collate<std::map<std::string, torch::Tensor>> {
public:
    explicit RolloutBufferBatchTransform(int64_t batchSize);
    ~RolloutBufferBatchTransform() override;
    OutputBatchType apply_batch(InputBatchType inputBatch) override;

private:
    //! The transition batch tensor mop for temporarily storing tensors before stacking them.
    std::map<std::string, std::vector<torch::Tensor>> transitionBatch_;
    static std::map<std::string, std::vector<torch::Tensor>> reserve_transition_batch_(int64_t batchSize);
};
/*!
 * @} @I{ // End group rollout_buffer_group }
 * @} @I{ // End group binaries_group }
 */

#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERBATCHTRANSFORM_H_
