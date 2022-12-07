
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

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup memory_group memory
 * @brief Memory module is the C++ backend for rlpack._C.memory.Memory class. Heavier workloads have been optimized
 * with multithreading with OpenMP and CUDA (if CUDA compatible device is found).
 * @{
 */
/*! @brief The class C_Memory is the C++ backend for memory-buffer used in algorithms that stores transitions in a buffer.
 * This class contains optimized routines to support Python front-end of rlpack._C.memory.Memory class.
 *
 * A `memory` index refers to an index that yields a transition from C_Memory. This works by indexing the following variables and grouping them together:
 *  - states_current: C_Memory::statesCurrent_;
 *  - states_next: C_Memory::statesNext_;
 *  - rewards: C_Memory::rewards_;
 *  - actions: C_Memory::actions_;
 *  - dones: C_Memory::dones_;
 *  - priorities: C_Memory::priorities_;
 *  - probabilities: C_Memory::probabilities_;
 *  - weights: C_Memory::weights_;
 *  Given index is hence applied to all the tensors.
 */
class C_Memory {

public:
    C_Memory();
    /*!
     * @brief The class C_MemoryData keeps the references to data that is associated with C_Memory. This class
     * implements the functions necessary to retrieve the data by de-referencing the data associated with C_Memory.
     */
    struct C_MemoryData {
        C_MemoryData();
        ~C_MemoryData();

    public:
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
        /*!
         * The map to store references to each deque that stores each quantity from transitions. This map stores the
         * references to following containers:
         *  - states_current: C_Memory::statesCurrent_;
         *  - states_next: C_Memory::statesNext_;
         *  - rewards: C_Memory::rewards_;
         *  - actions: C_Memory::actions_;
         *  - dones: C_Memory::dones_;
         *  - priorities: C_Memory::priorities_;
         *  - probabilities: C_Memory::probabilities_;
         *  - weights: C_Memory::weights_;
         */
        std::map<std::string, std::deque<torch::Tensor> *> transitionInformationReference_;
        //! The reference to deque that stores terminal state indices; C_Memory::terminalStateIndices_.
        std::deque<int64_t> *terminalIndicesReference_ = nullptr;
        //! The reference to deque that stores priorities float; C_Memory::prioritiesFloat_.
        std::deque<float_t> *prioritiesFloatReference_ = nullptr;
    };
    //! Shared Pointer to C_Memory::C_MemoryData.
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
                           torch::Tensor &newPriorities);
    [[nodiscard]] C_MemoryData view() const;
    void initialize(C_MemoryData &viewC_Memory);
    void clear();
    size_t size();
    int64_t num_terminal_states();
    int64_t tree_height();

private:
    //! Deque of torch tensors for current states.
    std::deque<torch::Tensor> statesCurrent_;
    //! Deque of torch tensors for next states.
    std::deque<torch::Tensor> statesNext_;
    //! Deque of torch tensors for rewards.
    std::deque<torch::Tensor> rewards_;
    //! Deque of torch tensors for actions.
    std::deque<torch::Tensor> actions_;
    //! Deque of torch tensors for dones.
    std::deque<torch::Tensor> dones_;
    //! Deque of torch tensors for priorities.
    std::deque<torch::Tensor> priorities_;
    //! Deque of torch tensors for probabilities.
    std::deque<torch::Tensor> probabilities_;
    //! Deque of torch tensors for weights.
    std::deque<torch::Tensor> weights_;
    //! Deque of integers indicating the indices of terminal states.
    std::deque<int64_t> terminalStateIndices_;
    //! Deque of float indicating the priorities in C++ float. Values are obtained from C_Memory::priorities_.
    std::deque<float_t> prioritiesFloat_;
    //! Vector of loaded indices. This indicates the indices that have been loaded out of total capacity of the memory.
    std::vector<int64_t> loadedIndices_;
    //! Shared Pointer to SumTree class object.
    std::shared_ptr<SumTree> sumTreeSharedPtr_;
    //! Torch device passed during class initialisation. Defaults to CPU.
    torch::Device device_ = torch::kCPU;
    //! Buffer size passed during the class initialisation. Defaults to 32768.
    int64_t bufferSize_ = 32768;
    //! The counter variable the tracks the loaded indices in sync with total timesteps. Once memory reaches the buffer size, this will not update.
    int64_t stepCounter_ = 0;
    //! The prioritization strategy code that is being. This determines the sampling technique that is employed. Refer rlpack.dqn.dqn.Dqn.get_prioritization_code.
    int32_t prioritizationStrategyCode_ = 0;
    //! The batch size that is set during class initialisation. Number of samples equivalent to this are selected during sampling.
    int32_t batchSize_ = 32;
    //! The map between std::string and torch::DeviceType; mapping the device name in string to DeviceType.
    std::map<std::string, torch::DeviceType> deviceMap_{
            {"cpu", torch::kCPU},
            {"cuda", torch::kCUDA},
            {"mps", torch::kMPS}};
    //! Offload class initialised with float template.
    Offload<float_t> *offloadFloat_;
    //! Offload class initialised with int64 template.
    Offload<int64_t> *offloadInt64_;
    //! The loaded indices slice; the slice of indices that is sampled during sampling process. In each sampling size its size is equal to C_Memory::batchSize_.
    std::vector<int64_t> loadedIndicesSlice_;
    //! The temporary buffer of loaded indices slice which is to be shuffled further before populating C_Memory::loadedIndicesSlice_.
    std::vector<int64_t> loadedIndicesSliceToShuffle_;
    //! The Quantile segment indices sampled when rank-based prioritization is used.
    std::vector<int64_t> segmentQuantileIndices_;
    //! The seed values generated during each sampling cycle for proportional based prioritization.
    std::vector<float_t> seedValues_;
    //! The sampled current state tensors from C_Memory::statesCurrent_.
    std::vector<torch::Tensor> sampledStateCurrent_;
    //! The sampled next state tensors from C_Memory::statesNext_.
    std::vector<torch::Tensor> sampledStateNext_;
    //! The sampled reward tensors from C_Memory::rewards_.
    std::vector<torch::Tensor> sampledRewards_;
    //! The sampled action tensors from C_Memory::actions_.
    std::vector<torch::Tensor> sampledActions_;
    //! The done tensors from C_Memory::dones_.
    std::vector<torch::Tensor> sampledDones_;
    //! The sampled priority tensors from C_Memory::priorities.
    std::vector<torch::Tensor> sampledPriorities_;
    //! The sampled indices as tensors from C_Memory::loadedIndices_.
    std::vector<torch::Tensor> sampledIndices_;

    static torch::Tensor compute_probabilities(torch::Tensor &priorities, float_t alpha);
    static torch::Tensor compute_important_sampling_weights(torch::Tensor &probabilities,
                                                            int64_t currentSize,
                                                            float_t beta);
};
/*!
 * @} @I{ // End group memory_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_MEMORY_C_MEMORY_CUH