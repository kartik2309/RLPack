//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#include "RolloutBufferBatch.h"

RolloutBufferBatch::RolloutBufferBatch(RolloutBufferContainer*& rolloutBufferContainer,
                                       torch::TensorOptions& tensorOptions) {
    rolloutBufferContainer_ = rolloutBufferContainer;
    tensorOptions_ = tensorOptions;
}

RolloutBufferBatch::~RolloutBufferBatch() = default;

RolloutBufferBatch::ExampleType RolloutBufferBatch::get(size_t index) {
    auto transition = rolloutBufferContainer_->transition_at(static_cast<int64_t>(index));
    return transition.get_transition_data(tensorOptions_);
}

c10::optional<size_t> RolloutBufferBatch::size() const {
    return rolloutBufferContainer_->size_transitions();
}