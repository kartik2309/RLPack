//
// Created by Kartik Rajeshwaran on 2022-12-27.
//

#include "C_ReplayBufferData.h"

/*!
 * The default constructor for C_MemoryData
 */
C_ReplayBufferData::C_ReplayBufferData() = default;

/*!
 * The default destructor for C_MemoryData
 */
C_ReplayBufferData::~C_ReplayBufferData() = default;

std::map<std::string, std::deque<torch::Tensor>> C_ReplayBufferData::dereference_transition_information() {
    /*!
     * The function to dereference the pointers from C_MemoryData::transitionInformationReference_.
     *
     * @return Map of string indicating the transition quantity name and the corresponding deque.
     */
    std::map<std::string, std::deque<torch::Tensor>> dereferencedTransitionInformation = {
            {"states_current", *transitionInformationReference_["states_current"]},
            {"states_next", *transitionInformationReference_["states_next"]},
            {"rewards", *transitionInformationReference_["rewards"]},
            {"actions", *transitionInformationReference_["actions"]},
            {"dones", *transitionInformationReference_["dones"]},
            {"priorities", *transitionInformationReference_["priorities"]},
            {"probabilities", *transitionInformationReference_["probabilities"]},
            {"weights", *transitionInformationReference_["weights"]},
    };

    return dereferencedTransitionInformation;
}

std::map<std::string, std::deque<int64_t>> C_ReplayBufferData::dereference_terminal_state_indices() const {
    /*!
     * The function to dereference the pointers from C_MemoryData::terminalIndicesReference_.
     *
     * @return The map between terminal state indices and the corresponding deque. Always returns the map with
     * key `terminal_state_indices`.
     */
    std::map<std::string, std::deque<int64_t>> dereferencedTerminalStates = {
            {"terminal_state_indices", *terminalIndicesReference_}};
    return dereferencedTerminalStates;
}

std::map<std::string, std::deque<float_t>> C_ReplayBufferData::dereference_priorities() const {
    /*!
     * The function to dereference the pointers from C_MemoryData::prioritiesFloatReference_.
     *
     * @return The map between float priorities and the corresponding deque. Always returns the map with
     * key `priorities`.
     */
    std::map<std::string, std::deque<float_t>> dereferencedPriorities = {
            {"priorities", *prioritiesFloatReference_}};
    return dereferencedPriorities;
}

void C_ReplayBufferData::set_transition_information_references(
        std::deque<torch::Tensor> *&statesCurrent,
        std::deque<torch::Tensor> *&statesNext,
        std::deque<torch::Tensor> *&rewards,
        std::deque<torch::Tensor> *&actions,
        std::deque<torch::Tensor> *&dones,
        std::deque<torch::Tensor> *&priorities,
        std::deque<torch::Tensor> *&probabilities,
        std::deque<torch::Tensor> *&weights) {
    /*!
     * Function to set the references to C_MemoryData::transitionInformationReference_.
     *
     * @param statesCurrent : The pointer to deque of current states; statesCurrent_.
     * @param statesNext : The pointer to deque of next states; statesNext_.
     * @param rewards : The pointer to deque of rewards; rewards_.
     * @param actions : The pointer to deque of actions; actions.
     * @param dones : The pointer to deque of dones; dones_.
     * @param priorities : The pointer to deque of priorities; priorities_.
     * @param probabilities : The pointer to deque of probabilities; probabilities_.
     * @param weights : The pointer to deque of weights; weights_.
     */
    transitionInformationReference_["states_current"] = statesCurrent;
    transitionInformationReference_["states_next"] = statesNext;
    transitionInformationReference_["rewards"] = rewards;
    transitionInformationReference_["actions"] = actions;
    transitionInformationReference_["dones"] = dones;
    transitionInformationReference_["priorities"] = priorities;
    transitionInformationReference_["probabilities"] = probabilities;
    transitionInformationReference_["weights"] = weights;
}

void C_ReplayBufferData::set_transition_information_references(
        std::string &key,
        std::deque<torch::Tensor> *&reference) {
    /*!
     * Function to set the references to C_MemoryData::transitionInformationReference_ for a single key.
     *
     * @param key : The key on which reference is to be set.
     * @param *reference : The reference pointer.
     */
    transitionInformationReference_[key] = reference;
}

void C_ReplayBufferData::set_terminal_state_indices_reference(
        std::deque<int64_t> *&terminalStateIndicesReference) {
    /*!
     * Function to set the references to C_MemoryData::transitionInformationReference_.
     *
     * @param *terminalStateIndicesReference: The reference to terminalStateIndices_.
     */
    terminalIndicesReference_ = terminalStateIndicesReference;
}

void C_ReplayBufferData::set_priorities_reference(
        std::deque<float_t> *&prioritiesFloatReference) {
    /*!
     * Function to set the references to C_MemoryData::prioritiesFloatReference_.
     *
     * @param prioritiesFloatReference: The reference to C_ReplayBuffer:prioritiesFloat_.
     */
    prioritiesFloatReference_ = prioritiesFloatReference;
}