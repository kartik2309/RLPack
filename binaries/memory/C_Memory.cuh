//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#ifndef RLPACK_BINARIES_MEMORY_C_MEMORY_CUH
#define RLPACK_BINARIES_MEMORY_C_MEMORY_CUH

#include <omp.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#include "sumtree/SumTree.h"
#include "utils/Utils.h"

class C_Memory {
    /*
   * The class C_Memory is the C++ backend for memory-buffer used in algorithms that stores transitions in a buffer.
   * This class contains optimized routines to support Python front-end of Memory class.
   */
public:
    C_Memory();
    struct C_MemoryData {
        /*
     * The class C_MemoryData keeps the references to data that is associated with C_Memory. This class
     * implements the functions necessary to retrieve the data by de-referencing the data associated with C_Memory.
     */
        C_MemoryData();
        ~C_MemoryData();

        std::map<std::string, std::deque<torch::Tensor>> dereference_transition_information();
        [[nodiscard]] std::map<std::string, std::deque<int64_t>> dereference_terminal_state_indices() const;
        [[nodiscard]] std::map<std::string, std::deque<float_t>> dereference_priorities() const;
        void set_transition_information_references(std::deque<torch::Tensor> *&statesCurrent,
                                                   std::deque<torch::Tensor> *&statesNext,
                                                   std::deque<torch::Tensor> *&rewards,
                                                   std::deque<torch::Tensor> *&actions,
                                                   std::deque<torch::Tensor> *&dones,
                                                   std::deque<torch::Tensor> *&priorities,
                                                   std::deque<torch::Tensor> *&probabilities,
                                                   std::deque<torch::Tensor> *&weights);
        void set_transition_information_references(std::string &key,
                                                   std::deque<torch::Tensor> *&reference);
        void set_terminal_state_indices_reference(std::deque<int64_t> *&terminalStateIndicesReference);
        void set_priorities_reference(std::deque<float_t> *&prioritiesFloatReference);

    private:
        std::map<std::string, std::deque<torch::Tensor> *> transitionInformationReference_;
        std::deque<int64_t> *terminalIndicesReference_ = nullptr;
        std::deque<float_t> *prioritiesFloatReference_ = nullptr;
    };
    std::shared_ptr<C_MemoryData> cMemoryData;

    explicit C_Memory(const pybind11::int_ &bufferSize,
                      const pybind11::str &device,
                      const pybind11::int_ &prioritizationStrategyCode,
                      const pybind11::int_ &batchSize);
    ~C_Memory();

    void insert(torch::Tensor &stateCurrent,
                torch::Tensor &stateNext,
                torch::Tensor &reward,
                torch::Tensor &action,
                torch::Tensor &done,
                torch::Tensor &priority,
                torch::Tensor &probability,
                torch::Tensor &weight,
                bool isTerminalState);
    std::map<std::string, torch::Tensor> get_item(int64_t index);
    void set_item(int64_t index,
                  torch::Tensor &stateCurrent,
                  torch::Tensor &stateNext,
                  torch::Tensor &reward,
                  torch::Tensor &action,
                  torch::Tensor &done,
                  torch::Tensor &priority,
                  torch::Tensor &probability,
                  torch::Tensor &weight,
                  bool isTerminalState);
    void delete_item(int64_t index);
    std::map<std::string, torch::Tensor> sample(float_t forceTerminalStateProbability,
                                                int64_t parallelismSizeThreshold,
                                                float_t alpha = 0.0,
                                                float_t beta = 0.0,
                                                int64_t numSegments = 0);
    void update_priorities(torch::Tensor &randomIndices,
                           torch::Tensor &newPriorities,
                           torch::Tensor &newProbabilities,
                           torch::Tensor &newWeights);
    [[nodiscard]] C_MemoryData view() const;
    void initialize(C_MemoryData &viewC_Memory);
    void clear();
    size_t size();
    int64_t num_terminal_states();
    int64_t tree_height();

private:
    std::deque<torch::Tensor> statesCurrent_;
    std::deque<torch::Tensor> statesNext_;
    std::deque<torch::Tensor> rewards_;
    std::deque<torch::Tensor> actions_;
    std::deque<torch::Tensor> dones_;
    std::deque<torch::Tensor> priorities_;
    std::deque<torch::Tensor> probabilities_;
    std::deque<torch::Tensor> weights_;
    std::deque<int64_t> terminalStateIndices_;
    std::deque<float_t> prioritiesFloat_;
    std::vector<int64_t> loadedIndices_;
    std::shared_ptr<SumTree> sumTreeSharedPtr_;
    torch::Device device_ = torch::kCPU;
    int64_t bufferSize_ = 32768;
    int64_t stepCounter_ = 0;
    int32_t prioritizationStrategyCode_ = 0;
    int32_t batchSize_ = 32;
    std::map<std::string, torch::DeviceType> deviceMap_{
            {"cpu", torch::kCPU},
            {"cuda", torch::kCUDA},
            {"mps", torch::kMPS}};
    Offload<float_t> *offloadFloat_;
    Offload<int64_t> *offloadInt64_;
    std::vector<int64_t> loadedIndicesSlice_, loadedIndicesSliceToShuffle_, segmentQuantileIndices_;
    std::vector<float_t> seedValues_;
    std::vector<torch::Tensor> sampledStateCurrent_, sampledStateNext_,
            sampledRewards_, sampledActions_,
            sampledDones_, sampledPriorities_,
            sampledIndices_;

    static torch::Tensor compute_probabilities(torch::Tensor &priorities, float_t alpha);
    static torch::Tensor compute_important_sampling_weights(torch::Tensor &probabilities,
                                                            int64_t currentSize,
                                                            float_t beta);
};

#endif//RLPACK_BINARIES_MEMORY_C_MEMORY_CUH