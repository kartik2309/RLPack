//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#include "RolloutBufferBatchTransform.h"

RolloutBufferBatchTransform::RolloutBufferBatchTransform(int64_t batchSize)
    : BatchLambda(nullptr) {
    /*!
     * Constructor for RolloutBufferBatchTransform. This will also allocate necessary memory preemptively.
     *
     * @param batchSize: The batch size to be used.
     */
    transitionBatch_ = reserve_transition_batch_(batchSize);
}

/*!
 * Default destructor for RolloutBufferBatchTransform.
 */
RolloutBufferBatchTransform::~RolloutBufferBatchTransform() = default;

RolloutBufferBatchTransform::OutputBatchType
RolloutBufferBatchTransform::apply_batch(RolloutBufferBatchTransform::InputBatchType inputBatch) {
    /*!
     * Applies transformation to the batch to obtain the batch in a single object. This method stacks the
     * transitions and returns the a single tensor map with stacked tensors.
     *
     * @param inputBatch: The input batch to be processed.
     */
    std::map<std::string, at::Tensor> stackedMap;
    for (uint64_t index = 0; index < inputBatch.size(); index++) {
        auto data = inputBatch[index];
        for (auto &mapData: data) {
            transitionBatch_[mapData.first][index] = mapData.second;
        }
    }
    for (auto &mapData: transitionBatch_) {
        stackedMap[mapData.first] = torch::stack(mapData.second);
    }
    return stackedMap;
}

std::map<std::string, std::vector<torch::Tensor>>
RolloutBufferBatchTransform::reserve_transition_batch_(int64_t batchSize) {
    /*!
     * Creates the tensors for transition in which tensors are kept temporarily before stacking them in
     * `apply_batch`. This will essentially allocate the memory for vector of tensors for batch size.
     */
    return {{STATE_CURRENT, std::vector<torch::Tensor>(batchSize)},
            {STATE_NEXT, std::vector<torch::Tensor>(batchSize)},
            {REWARD, std::vector<torch::Tensor>(batchSize)},
            {DONE, std::vector<torch::Tensor>(batchSize)}};
}
