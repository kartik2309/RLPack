//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#ifndef RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_
#define RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_

#include <torch/extension.h>
#include "../utils/maps.h"

class C_RolloutBuffer {

public:
    C_RolloutBuffer(int64_t bufferSize, std::string &device, std::string &dtype);
    ~C_RolloutBuffer();

    void insert(std::map<std::string, torch::Tensor> &inputMap);
    std::map<std::string, torch::Tensor> compute_returns(float_t gamma);
    std::map<std::string, torch::Tensor> get_stacked_rewards();
    std::map<std::string, torch::Tensor> get_stacked_action_probabilities();
    std::map<std::string, torch::Tensor> get_stacked_state_current_values();
    std::map<std::string, torch::Tensor> get_stacked_entropies();
    void clear();

private:
    int64_t bufferSize_;
    torch::DeviceType device_;
    torch::Dtype dtype_;
    torch::TensorOptions tensorOptions_;
    std::vector<torch::Tensor> rewards_;
    std::vector<torch::Tensor> actionLogProbabilities_;
    std::vector<torch::Tensor> stateCurrentValues_;
    std::vector<torch::Tensor> entropies_;
};


#endif//RLPACK_BINARIES_ROLLOUT_BUFFER_CROLLOUTBUFFER_H_
