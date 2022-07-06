//
// Created by Kartik Rajeshwaran on 2022-07-05.
//

#ifndef RLPACK_MEMORY_H
#define RLPACK_MEMORY_H

#include <torch/torch.h>

class Memory {

public:
    Memory();

    explicit Memory(int32_t bufferSize);

    ~Memory();

    void push_back(torch::Tensor &stateCurrent,
                   torch::Tensor &stateNext,
                   float reward,
                   int action,
                   int done);

    torch::Tensor stack_current_states();

    torch::Tensor stack_next_states();

    torch::Tensor stack_rewards();

    torch::Tensor stack_actions();

    torch::Tensor stack_dones();

    void clear();

    size_t size();

    void at(Memory *memory, int index);

    void reserve(int32_t bufferSize);

private:
    std::vector<torch::Tensor> stateCurrent_;
    std::vector<torch::Tensor> stateNext_;
    std::vector<torch::Tensor> reward_;
    std::vector<torch::Tensor> action_;
    std::vector<torch::Tensor> done_;
};

#endif //RLPACK_MEMORY_H
