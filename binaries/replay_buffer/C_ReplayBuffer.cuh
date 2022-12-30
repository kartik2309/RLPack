
#ifndef RLPACK_BINARIES_REPLAY_BUFFER_C_REPLAY_BUFFER_CUH
#define RLPACK_BINARIES_REPLAY_BUFFER_C_REPLAY_BUFFER_CUH

#include <omp.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#include "../utils/maps.h"
#include "offload/Offload.h"
#include "replay_buffer_data/C_ReplayBufferData.h"
#include "sumtree/SumTree.h"

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup replay_buffer_group replay_buffer
 * @brief The C++ backend for rlpack._C.replay_buffer.ReplayBuffer class. Heavier workloads have been optimized
 * with multithreading with OpenMP and CUDA (if CUDA compatible device is found).
 * @{
 */
/*! @brief The class C_ReplayBuffer is the C++ backend for replay buffer-buffer used in algorithms that stores transitions in a buffer.
 * This class contains optimized routines to support Python front-end of rlpack._C.replay_buffer.ReplayBuffer class.
 *
 * A `replay_buffer` index refers to an index that yields a transition from C_ReplayBuffer. This works by indexing the following variables and grouping them together:
 *  - states_current: C_ReplayBuffer::statesCurrent_;
 *  - states_next: C_ReplayBuffer::statesNext_;
 *  - rewards: C_ReplayBuffer::rewards_;
 *  - actions: C_ReplayBuffer::actions_;
 *  - dones: C_ReplayBuffer::dones_;
 *  - priorities: C_ReplayBuffer::priorities_;
 *  - probabilities: C_ReplayBuffer::probabilities_;
 *  - weights: C_ReplayBuffer::weights_;
 *  Given index is hence applied to all the tensors.
 */
class C_ReplayBuffer {

public:
    //! Shared Pointer to C_ReplayBuffer::C_MemoryData.
    std::shared_ptr<C_ReplayBufferData> cMemoryData;

    C_ReplayBuffer();
    explicit C_ReplayBuffer(int64_t bufferSize,
                            const std::string &device,
                            const std::string &dtype,
                            const int32_t &prioritizationStrategyCode,
                            const int32_t &batchSize);
    ~C_ReplayBuffer();

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
    [[nodiscard]] C_ReplayBufferData view() const;
    void initialize(C_ReplayBufferData &viewC_MemoryData);
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
    //! Deque of float indicating the priorities in C++ float. Values are obtained from C_ReplayBuffer::priorities_.
    std::deque<float_t> prioritiesFloat_;
    //! Vector of loaded indices. This indicates the indices that have been loaded out of total capacity of the replay buffer.
    std::vector<int64_t> loadedIndices_;
    //! Shared Pointer to SumTree class object.
    std::shared_ptr<SumTree> sumTreeSharedPtr_;
    //! Torch device passed during class initialisation. Defaults to CPU.
    torch::Device device_ = torch::kCPU;
    //! Torch datatype passed during class initialisation. Defaults to CPU.
    torch::Dtype dtype_ = torch::kFloat32;
    //! Buffer size passed during the class initialisation. Defaults to 32768.
    int64_t bufferSize_ = 32768;
    //! The counter variable the tracks the loaded indices in sync with total timesteps. Once replay buffer reaches the buffer size, this will not update.
    int64_t stepCounter_ = 0;
    //! The prioritization strategy code that is being. This determines the sampling technique that is employed. Refer rlpack.dqn.dqn.Dqn.get_prioritization_code.
    int32_t prioritizationStrategyCode_ = 0;
    //! The batch size that is set during class initialisation. Number of samples equivalent to this are selected during sampling.
    int32_t batchSize_ = 32;

    std::map<std::string, torch::DeviceType> deviceMap_{
            {"cpu", torch::kCPU},
            {"cuda", torch::kCUDA},
            {"mps", torch::kMPS}};
    //! Offload class initialised with float template.
    Offload<float_t> *offloadFloat_;
    //! Offload class initialised with int64 template.
    Offload<int64_t> *offloadInt64_;
    //! The loaded indices slice; the slice of indices that is sampled during sampling process. In each sampling size its size is equal to C_ReplayBuffer::batchSize_.
    std::vector<int64_t> loadedIndicesSlice_;
    //! The Quantile segment indices sampled when rank-based prioritization is used.
    std::vector<int64_t> segmentQuantileIndices_;
    //! The seed values generated during each sampling cycle for proportional based prioritization.
    std::vector<float_t> seedValues_;
    //! The sampled current state tensors from C_ReplayBuffer::statesCurrent_.
    std::vector<torch::Tensor> sampledStateCurrent_;
    //! The sampled next state tensors from C_ReplayBuffer::statesNext_.
    std::vector<torch::Tensor> sampledStateNext_;
    //! The sampled reward tensors from C_ReplayBuffer::rewards_.
    std::vector<torch::Tensor> sampledRewards_;
    //! The sampled action tensors from C_ReplayBuffer::actions_.
    std::vector<torch::Tensor> sampledActions_;
    //! The done tensors from C_ReplayBuffer::dones_.
    std::vector<torch::Tensor> sampledDones_;
    //! The sampled priority tensors from C_ReplayBuffer::priorities.
    std::vector<torch::Tensor> sampledPriorities_;
    //! The sampled indices as tensors from C_ReplayBuffer::loadedIndices_.
    std::vector<torch::Tensor> sampledIndices_;

    static torch::Tensor compute_probabilities(torch::Tensor &priorities, float_t alpha);
    static torch::Tensor compute_important_sampling_weights(torch::Tensor &probabilities,
                                                            int64_t currentSize,
                                                            float_t beta);
};
/*!
 * @} @I{ // End group replay_buffer_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_REPLAY_BUFFER_C_REPLAY_BUFFER_CUH