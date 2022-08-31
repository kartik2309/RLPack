//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#ifndef RLPACK_C_MEMORY_H
#define RLPACK_C_MEMORY_H

#include <pybind11/pybind11.h>
#include <random>
#include <omp.h>
#include <torch/extension.h>
#include <vector>
#include <deque>

class C_Memory {
public:

    struct C_MemoryData {
        std::map<std::string, std::deque<torch::Tensor> *> coreDataPtr;
        std::deque<int64_t> *terminalStatesIndicesPtr = nullptr;

        C_MemoryData();

        ~C_MemoryData();

        std::map<std::string, std::deque<torch::Tensor>> derefCoreData();

        [[nodiscard]] std::map<std::string, std::deque<int64_t>> derefTerminalStateIndices() const;
    } cMemoryData;

    explicit C_Memory(pybind11::int_ &bufferSize, pybind11::str &device);

    explicit C_Memory();

    ~C_Memory();

    void insert(torch::Tensor &stateCurrent, torch::Tensor &stateNext, torch::Tensor &reward,
                torch::Tensor &action, torch::Tensor &done, bool isTerminalState);

    std::map<std::string, torch::Tensor> get_item(int64_t index);

    void delete_item(int64_t index);

    std::map<std::string, torch::Tensor> sample(int32_t batchSize,
                                                float_t forceTerminalStateProbability,
                                                int64_t parallelismSizeThreshold);

    C_MemoryData view();

    void initialize(C_MemoryData &viewC_Memory);

    void clear();

    size_t size();

private:
    std::deque<torch::Tensor> statesCurrent_;
    std::deque<torch::Tensor> statesNext_;
    std::deque<torch::Tensor> rewards_;
    std::deque<torch::Tensor> actions_;
    std::deque<torch::Tensor> dones_;
    std::deque<int64_t> terminalStateIndices_;
    std::vector<int64_t> loadedIndices_;
    torch::Device device_ = torch::kCPU;
    int64_t bufferSize_ = 32768;
    int64_t step_counter_ = 0;
    std::map<std::string, torch::DeviceType> deviceMap{{"cpu",  torch::kCPU},
                                                       {"cuda", torch::kCUDA},
                                                       {"mps",  torch::kMPS}};

    std::vector<int64_t> shuffle_loaded_indices(int64_t parallelismSizeThreshold);

    static torch::Tensor adjust_dimensions(torch::Tensor &tensor, c10::IntArrayRef &targetDimensions);
};


#endif //RLPACK_C_MEMORY_H
