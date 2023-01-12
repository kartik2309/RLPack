//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERBATCHTRANSFORM_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERBATCHTRANSFORM_H_

#include <torch/extension.h>

#include "../rollout_buffer_data/RolloutBufferData.h"

class RolloutBufferBatchTransform
    : public torch::data::transforms::Collate<std::map<std::string, torch::Tensor>> {
public:
    explicit RolloutBufferBatchTransform(int64_t batchSize);
    ~RolloutBufferBatchTransform() override;
    OutputBatchType apply_batch(InputBatchType inputBatch) override;

private:
    std::map<std::string, std::vector<torch::Tensor>> transitionBatch_;
    static std::map<std::string, std::vector<torch::Tensor>> reserve_transition_batch(int64_t batchSize);
};

#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_CONTAINER_ROLLOUTBUFFERBATCHTRANSFORM_H_
