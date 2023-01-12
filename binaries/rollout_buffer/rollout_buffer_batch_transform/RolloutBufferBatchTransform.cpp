//
// Created by Kartik Rajeshwaran on 2023-01-11.
//

#include "RolloutBufferBatchTransform.h"

RolloutBufferBatchTransform::RolloutBufferBatchTransform(int64_t batchSize)
    : BatchLambda(nullptr) {
    transitionBatch_ = reserve_transition_batch(batchSize);
}

RolloutBufferBatchTransform::~RolloutBufferBatchTransform() = default;

RolloutBufferBatchTransform::OutputBatchType
RolloutBufferBatchTransform::apply_batch(RolloutBufferBatchTransform::InputBatchType inputBatch) {
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
RolloutBufferBatchTransform::reserve_transition_batch(int64_t batchSize) {
    return {{STATE_CURRENT, std::vector<torch::Tensor>(batchSize)},
            {STATE_NEXT, std::vector<torch::Tensor>(batchSize)},
            {REWARD, std::vector<torch::Tensor>(batchSize)},
            {DONE, std::vector<torch::Tensor>(batchSize)}};
}
