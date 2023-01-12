//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_BATCH_ROLLOUTBUFFERBATCH_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_BATCH_ROLLOUTBUFFERBATCH_H_

#include <torch/extension.h>

#include "../rollout_buffer_container/RolloutBufferContainer.h"

class RolloutBufferBatch
    : public torch::data::Dataset<RolloutBufferBatch, std::map<std::string, torch::Tensor>> {
public:
    RolloutBufferBatch(RolloutBufferContainer*& rolloutBufferContainer,
                       torch::TensorOptions& tensorOptions);
    ~RolloutBufferBatch() override;

    ExampleType get(size_t index) override;
    [[nodiscard]] c10::optional<size_t> size() const override;

private:
    RolloutBufferContainer* rolloutBufferContainer_;
    torch::TensorOptions tensorOptions_;
};



#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_ROLLOUT_BUFFER_BATCH_ROLLOUTBUFFERBATCH_H_
