//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#include "RolloutBufferBatch.h"

RolloutBufferBatch::RolloutBufferBatch(RolloutBufferContainer*& rolloutBufferContainer,
                                       torch::TensorOptions& tensorOptions) {
    /*!
     * The constructor RolloutBufferBatch.
     *
     * @param rolloutBufferContainer: The reference to the pointer to dynamically allocated object
     *  of RolloutBufferContainer.
     * @param tensorOptions: The tensor options for the batch.
     */
    rolloutBufferContainer_ = rolloutBufferContainer;
    tensorOptions_ = tensorOptions;
}

/*!
 * Default destructor for RolloutBufferBatch.
 */
RolloutBufferBatch::~RolloutBufferBatch() = default;

RolloutBufferBatch::ExampleType RolloutBufferBatch::get(size_t index) {
    /*!
     * Retrieves a transition sample.
     *
     * @param index: The index at which a transition sample is to be retrieved.
     * @return The transitions with each tensor moved and casted as tensorOptions_.
     */
    auto transition = rolloutBufferContainer_->transition_at(static_cast<int64_t>(index));
    return transition.get_transition_data(tensorOptions_);
}

c10::optional<size_t> RolloutBufferBatch::size() const {
    /*!
     * The size of the dataset. This corresponds to the number of transition samples.
     */
    return rolloutBufferContainer_->size_transitions();
}