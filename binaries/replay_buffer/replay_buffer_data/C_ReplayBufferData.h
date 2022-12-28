//
// Created by Kartik Rajeshwaran on 2022-12-27.
//

#ifndef RLPACK_BINARIES_REPLAY_BUFFER_REPLAY_BUFFER_DATA_CREPLAYBUFFERDATA_H_
#define RLPACK_BINARIES_REPLAY_BUFFER_REPLAY_BUFFER_DATA_CREPLAYBUFFERDATA_H_

#include <torch/extension.h>

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
/*!
 * @brief The class C_ReplayBuffer keeps the references to data that is associated with C_ReplayBuffer. This class
 * implements the functions necessary to retrieve the data by de-referencing the data associated with C_ReplayBuffer.
 */
class C_ReplayBufferData {

public:
    C_ReplayBufferData();
    ~C_ReplayBufferData();

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
     *  - states_current: C_ReplayBuffer::statesCurrent_;
     *  - states_next: C_ReplayBuffer::statesNext_;
     *  - rewards: C_ReplayBuffer::rewards_;
     *  - actions: C_ReplayBuffer::actions_;
     *  - dones: C_ReplayBuffer::dones_;
     *  - priorities: C_ReplayBuffer::priorities_;
     *  - probabilities: C_ReplayBuffer::probabilities_;
     *  - weights: C_ReplayBuffer::weights_;
     */
    std::map<std::string, std::deque<torch::Tensor> *> transitionInformationReference_;
    //! The reference to deque that stores terminal state indices; C_ReplayBuffer::terminalStateIndices_.
    std::deque<int64_t> *terminalIndicesReference_ = nullptr;
    //! The reference to deque that stores priorities float; C_ReplayBuffer::prioritiesFloat_.
    std::deque<float_t> *prioritiesFloatReference_ = nullptr;
};
/*!
 * @} @I{ // End group replay_buffer_group }
 * @} @I{ // End group binaries_group }
 */

#endif//RLPACK_BINARIES_REPLAY_BUFFER_REPLAY_BUFFER_DATA_CREPLAYBUFFERDATA_H_
