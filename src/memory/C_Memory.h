//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#ifndef RLPACK_C_MEMORY_H
#define RLPACK_C_MEMORY_H

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <optional>
#include <random>
#include <iostream>
#include <omp.h>
#include <torch/extension.h>

class C_Memory {
public:

    struct C_MemoryData {
        std::map<std::string, std::vector<torch::Tensor> *> coreDataPtr;
        std::vector<int64_t> *terminalStatesIndicesPtr = nullptr;

        C_MemoryData();

        ~C_MemoryData();

        std::map<std::string, std::vector<torch::Tensor>> derefCoreData();

        [[nodiscard]] std::vector<int64_t> derefTerminalStateIndices() const;
    } cMemoryData;

    explicit C_Memory(pybind11::int_ &bufferSize, pybind11::str &device);

    explicit C_Memory();

    ~C_Memory();

    void insert(torch::Tensor &stateCurrent, torch::Tensor &stateNext, torch::Tensor &reward,
                torch::Tensor &action, torch::Tensor &done, bool isTerminalState);

    void reserve(int64_t bufferSize);

    std::vector<torch::Tensor> get_item(int64_t index);

    void delete_item(int64_t index);

    std::map<std::string, torch::Tensor> sample(int32_t batchSize, float_t forceTerminalStateProbability);

    std::map<std::string, torch::Tensor> sample(int32_t batchSize);

    C_MemoryData view();

    void initialize(C_MemoryData &viewC_Memory);

    void clear();

    size_t size();

private:
    std::vector<torch::Tensor> statesCurrent_;
    std::vector<torch::Tensor> statesNext_;
    std::vector<torch::Tensor> rewards_;
    std::vector<torch::Tensor> actions_;
    std::vector<torch::Tensor> dones_;
    std::vector<int64_t> terminalStateIndices_;
    std::vector<int64_t> loadedIndices_;
    torch::Device device_ = torch::kCPU;
    int64_t bufferSize_ = 32768;
    int64_t step_counter_ = 0;
    std::map<std::string, torch::DeviceType> deviceMap{{"cpu",  torch::kCPU},
                                                       {"cuda", torch::kCUDA},
                                                       {"mps",  torch::kMPS}};

    std::vector<int64_t> shuffle_loaded_indices();
};


#endif //RLPACK_C_MEMORY_H
