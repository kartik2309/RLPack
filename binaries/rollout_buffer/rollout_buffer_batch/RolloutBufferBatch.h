//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_BATCH_ROLLOUTBUFFERBATCH_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_BATCH_ROLLOUTBUFFERBATCH_H_

#include <torch/extension.h>

#include "../rollout_buffer_container/RolloutBufferContainer.h"

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
  * @brief The class RolloutBufferBatch creates a PyTorch dataset. Current implementation creates the dataset for
  * transitions.
  */
class RolloutBufferBatch
    : public torch::data::Dataset<RolloutBufferBatch, std::map<std::string, torch::Tensor>> {
public:
    RolloutBufferBatch(RolloutBufferContainer*& rolloutBufferContainer,
                       torch::TensorOptions& tensorOptions);
    ~RolloutBufferBatch() override;

    ExampleType get(size_t index) override;
    [[nodiscard]] c10::optional<size_t> size() const override;

private:
    //! The input `rolloutBufferContainer` from class constructor; a pointer to RolloutBufferContainer object.
    RolloutBufferContainer* rolloutBufferContainer_;
    //! The input `tensorOptions` from class constructor; the tensor options to move/cast the tensors.
    torch::TensorOptions tensorOptions_;
};
/*!
 * @} @I{ // End group rollout_buffer_group }
 * @} @I{ // End group binaries_group }
 */



#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_BATCH_ROLLOUTBUFFERBATCH_H_
