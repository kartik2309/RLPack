
#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
#include "C_Memory.cuh"

/*!
 * The default constructor for C_Memory::C_MemoryData
 */
C_Memory::C_MemoryData::C_MemoryData() = default;

/*!
 * The default destructor for C_Memory::C_MemoryData
 */
C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::deque<torch::Tensor>> C_Memory::C_MemoryData::dereference_transition_information() {
    /*!
     * The function to dereference the pointers from C_Memory::C_MemoryData::transitionInformationReference_.
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

std::map<std::string, std::deque<int64_t>> C_Memory::C_MemoryData::dereference_terminal_state_indices() const {
    /*!
     * The function to dereference the pointers from C_Memory::C_MemoryData::terminalIndicesReference_.
     *
     * @return The map between terminal state indices and the corresponding deque. Always returns the map with
     * key `terminal_state_indices`.
     */
    std::map<std::string, std::deque<int64_t>> dereferencedTerminalStates = {
            {"terminal_state_indices", *terminalIndicesReference_}};
    return dereferencedTerminalStates;
}

std::map<std::string, std::deque<float_t>> C_Memory::C_MemoryData::dereference_priorities() const {
    /*!
     * The function to dereference the pointers from C_Memory::C_MemoryData::prioritiesFloatReference_.
     *
     * @return The map between float priorities and the corresponding deque. Always returns the map with
     * key `priorities`.
     */
    std::map<std::string, std::deque<float_t>> dereferencedPriorities = {
            {"priorities", *prioritiesFloatReference_}};
    return dereferencedPriorities;
}

void C_Memory::C_MemoryData::set_transition_information_references(
        std::deque<torch::Tensor> *&statesCurrent,
        std::deque<torch::Tensor> *&statesNext,
        std::deque<torch::Tensor> *&rewards,
        std::deque<torch::Tensor> *&actions,
        std::deque<torch::Tensor> *&dones,
        std::deque<torch::Tensor> *&priorities,
        std::deque<torch::Tensor> *&probabilities,
        std::deque<torch::Tensor> *&weights) {
    /*!
     * Function to set the references to C_Memory::C_MemoryData::transitionInformationReference_.
     *
     * @param statesCurrent : The pointer to deque of current states; C_Memory::statesCurrent_.
     * @param statesNext : The pointer to deque of next states; C_Memory::statesNext_.
     * @param rewards : The pointer to deque of rewards; C_Memory::rewards_.
     * @param actions : The pointer to deque of actions; C_Memory::actions.
     * @param dones : The pointer to deque of dones; C_Memory::dones_.
     * @param priorities : The pointer to deque of priorities; C_Memory::priorities_.
     * @param probabilities : The pointer to deque of probabilities; C_Memory::probabilities_.
     * @param weights : The pointer to deque of weights; C_Memory::weights_.
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

void C_Memory::C_MemoryData::set_transition_information_references(
        std::string &key,
        std::deque<torch::Tensor> *&reference) {
    /*!
     * Function to set the references to C_Memory::C_MemoryData::transitionInformationReference_ for a single key.
     *
     * @param key : The key on which reference is to be set.
     * @param *reference : The reference pointer.
     */
    transitionInformationReference_[key] = reference;
}

void C_Memory::C_MemoryData::set_terminal_state_indices_reference(
        std::deque<int64_t> *&terminalStateIndicesReference) {
    /*!
     * Function to set the references to C_Memory::C_MemoryData::transitionInformationReference_.
     *
     * @param *terminalStateIndicesReference: The reference to C_Memory::terminalStateIndices_.
     */
    terminalIndicesReference_ = terminalStateIndicesReference;
}

void C_Memory::C_MemoryData::set_priorities_reference(
        std::deque<float_t> *&prioritiesFloatReference) {
    /*!
     * Function to set the references to C_Memory::C_MemoryData::prioritiesFloatReference_.
     *
     * @param prioritiesFloatReference: The reference to C_Memory:prioritiesFloat_.
     */
    prioritiesFloatReference_ = prioritiesFloatReference;
}

C_Memory::C_Memory(const pybind11::int_ &bufferSize,
                   const pybind11::str &device,
                   const pybind11::int_ &prioritizationStrategyCode,
                   const pybind11::int_ &batchSize) {
    /*!
     * The class constructor for C_Memory. This constructor initialised the C_Memory class and allocates the
     * required memory as per input arguments. This initialises the rlpack._C.memory.Memory.c_memory and is
     * equivalent to rlpack._C.memory.Memory.__init__.
     *
     * @param bufferSize : The buffer size to be used and allocated for the memory.
     * @param device : The device transfer relevant tensors to.
     * @param prioritizationStrategyCode : The prioritization strategy code. Refer
     *  rlpack.dqn.dqn.Dqn.get_prioritization_code.
     * @param batchSize : The batch size to be used for sampling.
     *
     */
    bufferSize_ = bufferSize.cast<int64_t>();
    device_ = deviceMap_[device.cast<std::string>()];
    prioritizationStrategyCode_ = prioritizationStrategyCode.cast<int32_t>();
    batchSize_ = batchSize.cast<int32_t>();
    cMemoryData = std::make_shared<C_MemoryData>();
    auto statesCurrentRawPointer = &statesCurrent_;
    auto statesNextRawPointer = &statesNext_;
    auto rewardsRawPointer = &rewards_;
    auto actionsRawPointer = &actions_;
    auto donesRawPointer = &dones_;
    auto prioritiesRawPointer = &priorities_;
    auto probabilitiesRawPointer = &probabilities_;
    auto weightsRawPointer = &weights_;
    cMemoryData->set_transition_information_references(statesCurrentRawPointer,
                                                       statesNextRawPointer,
                                                       rewardsRawPointer,
                                                       actionsRawPointer,
                                                       donesRawPointer,
                                                       prioritiesRawPointer,
                                                       probabilitiesRawPointer,
                                                       weightsRawPointer);
    auto terminalStateIndicesRawPointer = &terminalStateIndices_;
    cMemoryData->set_terminal_state_indices_reference(terminalStateIndicesRawPointer);
    auto prioritiesFloatRawPointer = &prioritiesFloat_;
    cMemoryData->set_priorities_reference(prioritiesFloatRawPointer);
    loadedIndices_.reserve(bufferSize_);
    sumTreeSharedPtr_ = nullptr;
    switch (prioritizationStrategyCode_) {
        case 1:
            sumTreeSharedPtr_ = std::make_shared<SumTree>(bufferSize_);
            loadedIndicesSliceToShuffle_ = std::vector<int64_t>(batchSize_);
            break;
        case 2:
            loadedIndicesSliceToShuffle_ = std::vector<int64_t>(batchSize_);
            segmentQuantileIndices_ = std::vector<int64_t>(batchSize_);
            break;
        default:
            break;
    }
    offloadFloat_ = new Offload<float_t>(bufferSize_);
    offloadInt64_ = new Offload<int64_t>(bufferSize_);
    loadedIndicesSlice_ = std::vector<int64_t>(batchSize_);
    seedValues_ = std::vector<float_t>(bufferSize_);
    sampledStateCurrent_ = std::vector<torch::Tensor>(batchSize_);
    sampledStateNext_ = std::vector<torch::Tensor>(batchSize_);
    sampledRewards_ = std::vector<torch::Tensor>(batchSize_);
    sampledActions_ = std::vector<torch::Tensor>(batchSize_);
    sampledDones_ = std::vector<torch::Tensor>(batchSize_);
    sampledPriorities_ = std::vector<torch::Tensor>(batchSize_);
    sampledIndices_ = std::vector<torch::Tensor>(batchSize_);
}

C_Memory::C_Memory() {
    /*!
     * The default non-parameterised constructor. This constructor allocates memory as per default initialised variables.
     * This initialises the rlpack._C.memory.Memory.c_memory and is equivalent to rlpack._C.memory.Memory.__init__.
     */
    cMemoryData = std::make_shared<C_MemoryData>();
    auto statesCurrentRawPointer = &statesCurrent_;
    auto statesNextRawPointer = &statesNext_;
    auto rewardsRawPointer = &rewards_;
    auto actionsRawPointer = &actions_;
    auto donesRawPointer = &dones_;
    auto prioritiesRawPointer = &priorities_;
    auto probabilitiesRawPointer = &probabilities_;
    auto weightsRawPointer = &weights_;
    cMemoryData->set_transition_information_references(statesCurrentRawPointer,
                                                       statesNextRawPointer,
                                                       rewardsRawPointer,
                                                       actionsRawPointer,
                                                       donesRawPointer,
                                                       prioritiesRawPointer,
                                                       probabilitiesRawPointer,
                                                       weightsRawPointer);
    auto terminalStateIndicesRawPointer = &terminalStateIndices_;
    cMemoryData->set_terminal_state_indices_reference(terminalStateIndicesRawPointer);
    auto prioritiesFloatRawPointer = &prioritiesFloat_;
    cMemoryData->set_priorities_reference(prioritiesFloatRawPointer);
    loadedIndices_.reserve(bufferSize_);
    sumTreeSharedPtr_ = nullptr;
    switch (prioritizationStrategyCode_) {
        case 1:
            sumTreeSharedPtr_ = std::make_shared<SumTree>(bufferSize_);
            loadedIndicesSliceToShuffle_ = std::vector<int64_t>(batchSize_);
            break;
        case 2:
            loadedIndicesSliceToShuffle_ = std::vector<int64_t>(batchSize_);
            segmentQuantileIndices_ = std::vector<int64_t>(batchSize_);
            break;
        default:
            break;
    }
    offloadFloat_ = new Offload<float_t>(bufferSize_);
    offloadInt64_ = new Offload<int64_t>(bufferSize_);
    loadedIndicesSlice_ = std::vector<int64_t>(batchSize_);
    seedValues_ = std::vector<float_t>(bufferSize_);
}


C_Memory::~C_Memory() {
    /*!
     * The destructor for C_Memory.
     */
    delete offloadFloat_;
    delete offloadInt64_;
}

void C_Memory::insert(torch::Tensor &stateCurrent,
                      torch::Tensor &stateNext,
                      torch::Tensor &reward,
                      torch::Tensor &action,
                      torch::Tensor &done,
                      torch::Tensor &priority,
                      torch::Tensor &probability,
                      torch::Tensor &weight,
                      bool isTerminalState) {
    /*!
     * Insertion method for C_Memory. This is the C++ backend of rlpack._C.memory.Memory.insert method.
     *
     * @param stateCurrent : Current state from transition
     * @param stateNext : Next state from transition.
     * @param reward : Reward obtained during transition.
     * @param action : Action taken during transition.
     * @param done : Flag indicating if next state is terminal packaged in PyTorch Tensor.
     * @param priority : Priority value associated with the transition.
     * @param probability : Probability value associated with the transition.
     * @param weight : Weight value associated with the transition.
     * @param isTerminalState : Flag indicating if next state is terminal.
     */
    if (size() > bufferSize_) {
        delete_item(0);
    }
    statesCurrent_.push_back(stateCurrent);
    statesNext_.push_back(stateNext);
    rewards_.push_back(reward);
    actions_.push_back(action);
    dones_.push_back(done);
    priorities_.push_back(priority);
    probabilities_.push_back(probability);
    weights_.push_back(weight);
    prioritiesFloat_.push_back(priority.item<float_t>());
    if (size() < bufferSize_) {
        loadedIndices_.push_back(stepCounter_);
        stepCounter_ += 1;
    }
    if (isTerminalState) {
        terminalStateIndices_.push_back((int64_t) size() - 1);
    }
}

std::map<std::string, torch::Tensor> C_Memory::get_item(int64_t index) {
    /*!
     * Getter method for C_Memory. This is the C++ backend of rlpack._C.memory.Memory.__getitem__ method so can be
     * accessed by simple indexing operation (with operator []; item = memory[index]) from Python side.
     *
     * @param index : The index from which we want to obtain the transition
     * @return A map of transition quantities. The map will contain the following keys:
     *  - states_current
     *  - states_next
     *  - rewards
     *  - actions
     *  - dones
     *  - priorities
     *  - probabilities
     *  - weights
     */
    if (index >= size()) {
        throw std::out_of_range("Index is larger than current size of memory!");
    }
    std::map<std::string, torch::Tensor> returnItems = {
            {"states_current", statesCurrent_[index]},
            {"states_next", statesNext_[index]},
            {"rewards", rewards_[index]},
            {"actions", actions_[index]},
            {"dones", dones_[index]},
            {"priorities", priorities_[index]},
            {"probabilities", probabilities_[index]},
            {"weights", weights_[index]},
    };
    return returnItems;
}

void C_Memory::set_item(int64_t index,
                        torch::Tensor &stateCurrent,
                        torch::Tensor &stateNext,
                        torch::Tensor &reward,
                        torch::Tensor &action,
                        torch::Tensor &done,
                        torch::Tensor &priority,
                        torch::Tensor &probability,
                        torch::Tensor &weight,
                        bool isTerminalState) {
    /*!
     * Setter method for C_Memory. This is the C++ backend of rlpack._C.memory.Memory.__setitem__ method so can be
     * accessed by simple indexing operation (with operator []; memory[index] = index) from Python side.
     * This method modified the items at the given index.
     *
     * @param index : The index to which we want to set the transition.
     * @param stateCurrent : Current state from transition
     * @param stateNext : Next state from transition.
     * @param reward : Reward obtained during transition.
     * @param action : Action taken during transition.
     * @param done : Flag indicating if next state is terminal packaged in PyTorch Tensor.
     * @param priority : Priority value associated with the transition.
     * @param probability : Probability value associated with the transition.
     * @param weight : Weight value associated with the transition.
     * @param isTerminalState : Flag indicating if next state is terminal.
     */
    if (index >= size()) {
        throw std::out_of_range("Given index is larger than current size! Use insert method to expand the memory.");
    }
    statesCurrent_[index] = stateCurrent;
    statesNext_[index] = stateNext;
    rewards_[index] = reward;
    actions_[index] = action;
    dones_[index] = done;
    priorities_[index] = priority;
    probabilities_[index] = probability;
    weights_[index] = weight;
    if (isTerminalState) {
        auto findIter = std::find(
                terminalStateIndices_.begin(), terminalStateIndices_.end(), index);
        if (findIter != terminalStateIndices_.end()) {
            terminalStateIndices_.push_back(index);
        }
    }
    prioritiesFloat_[index] = priority.item<float_t>();
    loadedIndices_[index] = index;
    switch (prioritizationStrategyCode_) {
        case 1:
            sumTreeSharedPtr_->update(index, priority.item<float_t>());
        default:
            break;
    }
}

void C_Memory::delete_item(int64_t index) {
    /*!
     * Deletion method for C_Memory. This is the C++ backend of rlpack._C.memory.Memory.__delitem__ so can be
     * accessed by simple indexing operation (with operator []; del memory[index]) from Python side.
     *
     *
     * This the deletion is fast if index is either the first or last element, else will take O(n) to allocate memory
     * for items after index.
     *
     * @param index : The index of the transition we want to remove.
     */
    if (index >= size()) {
        throw std::out_of_range("Index is larger than current size of memory!");
    }
    if (index != 0) {
        statesCurrent_.erase(statesCurrent_.begin() + index);
        statesNext_.erase(statesNext_.begin() + index);
        rewards_.erase(rewards_.begin() + index);
        actions_.erase(actions_.begin() + index);
        if (dones_[index].flatten().item<int32_t>() == 1) {
            auto indexIter = std::find(
                    terminalStateIndices_.begin(), terminalStateIndices_.end(), index);
            if (indexIter != terminalStateIndices_.end()) {
                terminalStateIndices_.erase(indexIter);
            } else {
                std::cerr << "WARNING: Deletion of terminal state occurred but "
                             "terminal state was not found in `terminalStateIndices_`"
                          << std::endl;
            }
        }
        dones_.erase(dones_.begin() + index);
        priorities_.erase(priorities_.begin() + index);
        probabilities_.erase(probabilities_.begin() + index);
        weights_.erase(weights_.begin() + index);
        prioritiesFloat_.erase(prioritiesFloat_.begin() + index);
    } else {
        statesCurrent_.pop_front();
        statesNext_.pop_front();
        rewards_.pop_front();
        actions_.pop_front();
        dones_.pop_front();
        priorities_.pop_front();
        probabilities_.pop_front();
        weights_.pop_front();
        if (dones_[0].flatten().item<int32_t>() == 1) {
            terminalStateIndices_.pop_front();
        }
        prioritiesFloat_.pop_front();
    }
}

std::map<std::string, torch::Tensor> C_Memory::sample(float_t forceTerminalStateProbability,
                                                      int64_t parallelismSizeThreshold,
                                                      float_t alpha,
                                                      float_t beta,
                                                      int64_t numSegments) {
    /*!
     * The sampling method for C_Memory. This is the C++ backend of rlpack._C.memory.Memory.sample. Sampling is done
     * as per the prioritization strategy specified during initialisation of C_Memory.
     *
     * @param forceTerminalStateProbability : The probability to force a terminal state in final sample.
     * @param parallelismSizeThreshold : The threshold size of buffer (from C_Memory::size method) beyond with
     * OpenMP parallelized routines are used for sampling.
     * @param alpha : The alpha value for prioritization. This is used to compute probabilities, where higher alpha
     * indicates more aggressive prioritization.
     * @param beta : The beta value for prioritization. This is used to compute important sampling weights, where higher
     * beta indicates more aggressive bias correction.
     * @param numSegments : The number of segments to be used for rank-based prioritization (in accordance with Zipf's law)
     * @return A map of sampled transitions separated by quantities. The map has the following keys with
     * each key containing a tensor of shape `(batchSize, ...)`:
     *  - states_current
     *  - states_next
     *  - rewards
     *  - actions
     *  - dones
     *  - priorities
     *  - probabilities
     *  - weights
     */
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float_t> distributionP(0, 1);
    std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                       (int64_t) terminalStateIndices_.size() - 1);

    int64_t index = 0;
    bool forceTerminalState = false;
    switch (prioritizationStrategyCode_) {
        case 0: {
            offloadInt64_->reset();
            offloadInt64_->shuffle(loadedIndices_, parallelismSizeThreshold);
            memcpy(&loadedIndicesSlice_[0], &offloadInt64_->result[0], sizeof(int64_t) * batchSize_);
            break;
        }
        case 1: {
            // Proportional prioritization sampling.
            {
                bool enableParallelism = parallelismSizeThreshold < prioritiesFloat_.size();
//        Parallel region to execute resetting and creation of tree simultaneously with getting priority seeds.
//        Two sections are spawned for each associated function.
//        Sections 0: Executes resetting and creation of tree.
//        Sections 1: Executes computation of cumulative sum and generates priority seeds.
#pragma omp parallel sections if (enableParallelism) default(none) \
        firstprivate(parallelismSizeThreshold, prioritiesFloat_)   \
                shared(sumTreeSharedPtr_, seedValues_)
                {
#pragma omp section
                    {
                        sumTreeSharedPtr_->reset(parallelismSizeThreshold);
                        std::optional<std::vector<SumTreeNode *>> nullOptVector = std::nullopt;
                        sumTreeSharedPtr_->create_tree(prioritiesFloat_, nullOptVector);
                    }
#pragma omp section
                    {
                        offloadFloat_->reset();
                        auto cumulativeSum = offloadFloat_->cumulative_sum(prioritiesFloat_,
                                                                           parallelismSizeThreshold);
                        if (seedValues_.size() < static_cast<size_t>(cumulativeSum)) {
                            seedValues_.resize(static_cast<size_t>(cumulativeSum));
                        }
                        offloadFloat_->reset();
                        offloadFloat_->generate_priority_seeds(cumulativeSum, parallelismSizeThreshold);
                        memcpy(&seedValues_[0], &offloadFloat_->result[0], sizeof(float_t) * batchSize_);
                    }
                }
            }
            for (int32_t batchIndex = 0; batchIndex < batchSize_; batchIndex++) {
                auto seedValue = seedValues_[batchIndex];
                auto randomIndex = sumTreeSharedPtr_->sample(seedValue,
                                                             (int64_t) size());
                loadedIndicesSlice_[batchIndex] = randomIndex;
            }
            break;
        }
        case 2: {
            // Rank-Based prioritization sampling.
            int64_t previousQuantileIndex = 0, generatedRandomIndex = 0;
            index = 0;
            std::uniform_int_distribution<int64_t> distributionOfSegment(-1, 1);
            offloadFloat_->reset();
            offloadFloat_->arg_quantile_segment_indices(numSegments,
                                                        prioritiesFloat_,
                                                        parallelismSizeThreshold);
            memcpy(&segmentQuantileIndices_[0], &offloadFloat_->result[0], sizeof(float_t) * batchSize_);
            for (auto &segmentQuantileIndex: segmentQuantileIndices_) {
                if ((segmentQuantileIndex - previousQuantileIndex - 1) <= 1) {
                    generatedRandomIndex = previousQuantileIndex;
                } else {
                    distributionOfSegment.reset();
                    distributionOfSegment.param(std::uniform_int_distribution<int64_t>::param_type(previousQuantileIndex,
                                                                                                   segmentQuantileIndex - 1));
                    generatedRandomIndex = distributionOfSegment(generator);
                }
                loadedIndicesSlice_[index] = loadedIndices_[generatedRandomIndex];
                previousQuantileIndex = segmentQuantileIndex;
                index++;
            }
            index = 0;
            break;
        }
        default:
            break;
    }
    float_t p = distributionP(generator);
    if (size() < batchSize_) {
        throw std::out_of_range("Requested batch size is larger than current Memory size!");
    }
    if (p < forceTerminalStateProbability && terminalStateIndices_.size() > 1) {
        forceTerminalState = true;
    }
    if (forceTerminalState) {
        int64_t randomIndexToInsertTerminalState = distributionOfTerminalIndex(generator) % batchSize_;
        int64_t randomTerminalStateInfoIndex = distributionOfTerminalIndex(generator);
        int64_t randomTerminalStateIndex = terminalStateIndices_[randomTerminalStateInfoIndex];
        loadedIndicesSlice_[randomIndexToInsertTerminalState] = randomTerminalStateIndex;
    }
    for (auto &loadedIndex: loadedIndicesSlice_) {
        sampledStateCurrent_[index] = statesCurrent_[loadedIndex];
        sampledStateNext_[index] = statesNext_[loadedIndex];
        sampledRewards_[index] = rewards_[loadedIndex];
        sampledActions_[index] = actions_[loadedIndex];
        sampledDones_[index] = dones_[loadedIndex];
        sampledPriorities_[index] = priorities_[loadedIndex];
        sampledIndices_[index] = torch::full({}, loadedIndex);
        index++;
    }
    auto floatTensorOptions = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    auto int64TensorOptions = torch::TensorOptions().device(device_).dtype(torch::kInt64);
    auto statesCurrentStacked = torch::stack(sampledStateCurrent_, 0).to(floatTensorOptions);
    auto statesNextStacked = torch::stack(sampledStateNext_, 0).to(floatTensorOptions);
    auto rewardsStacked = torch::stack(sampledRewards_, 0).to(floatTensorOptions);
    auto actionsStacked = torch::stack(sampledActions_, 0).to(int64TensorOptions);
    auto donesStacked = torch::stack(sampledDones_, 0).to(floatTensorOptions);
    auto prioritiesStacked = torch::stack(sampledPriorities_, 0).to(floatTensorOptions);
    auto sampledIndicesStacked = torch::stack(sampledIndices_, 0).to(int64TensorOptions);
    std::map<std::string, torch::Tensor> samples = {
            {"states_current", statesCurrentStacked},
            {"states_next", statesNextStacked},
            {"rewards", rewardsStacked},
            {"actions", actionsStacked},
            {"dones", donesStacked},
            {"priorities", prioritiesStacked},
            {"random_indices", sampledIndicesStacked}};
    if (prioritizationStrategyCode_ != 0) {
        auto probabilities = compute_probabilities(prioritiesStacked, alpha).to(floatTensorOptions);
        auto weights = compute_important_sampling_weights(probabilities,
                                                          (int64_t) size(),
                                                          beta)
                               .to(floatTensorOptions);
        samples["probabilities"] = probabilities;
        samples["weights"] = weights;
    } else {
        samples["probabilities"] = torch::zeros(prioritiesStacked.sizes(), floatTensorOptions);
        samples["weights"] = torch::zeros(prioritiesStacked.sizes(), floatTensorOptions);
    }
    return samples;
}

void C_Memory::update_priorities(torch::Tensor &randomIndices,
                                 torch::Tensor &newPriorities) {
    /*!
     * The method to update priorities as per new values computed by agent as per the prioritization strategy. This
     * is the C++ backend of rlpack._C.memory.Memory.update_priorities method.
     *
     * @param randomIndices : The random indices on which priorities are required to be updated. C_Memory::sample
     * provides this information which can be used.
     * @param newPriorities : The new priorities computed by the agent as per the prioritization strategy.
     */
    if (prioritizationStrategyCode_ == 0) {
        throw std::runtime_error("`update_priorities` method called in C++ backend when C_Memory is un-prioritized!");
    }
    newPriorities = newPriorities.flatten();
    randomIndices = randomIndices.flatten();
    auto size = randomIndices.size(0);
    for (int32_t index = 0; index < size; index++) {
        auto selectIndex = randomIndices[index].item<int64_t>();
        priorities_[selectIndex] = newPriorities[index];
        prioritiesFloat_[selectIndex] = newPriorities[index].item<float_t>();
    }
}

C_Memory::C_MemoryData C_Memory::view() const {
    /*!
     * The pointer to C_Memory::C_MemoryData object. This will contain references of data in C_Memory and provides
     * an easy data view. This is the C++ backend of rlpack._C.memory.Memory.view method.
     */
    return *cMemoryData;
}

void C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
    /*!
     * Initialize method for C_Memory for initializing all the data from an object of C_Memory::C_MemoryData. This is
     * the C++ backend of rlpack._C.memory.Memory.initialize method
     *
     * @param viewC_MemoryData : An object of C_Memory::C_MemoryData.
     */
    cMemoryData = std::make_shared<C_MemoryData>(viewC_MemoryData);
    auto transitionInformation = cMemoryData->dereference_transition_information();
    auto terminalStateIndices = cMemoryData->dereference_terminal_state_indices();
    auto prioritiesFloat = cMemoryData->dereference_priorities();
    statesCurrent_ = transitionInformation["states_current"];
    statesNext_ = transitionInformation["states_next"];
    rewards_ = transitionInformation["rewards"];
    actions_ = transitionInformation["actions"];
    dones_ = transitionInformation["dones"];
    priorities_ = transitionInformation["priorities"];
    probabilities_ = transitionInformation["probabilities"];
    weights_ = transitionInformation["weights"];
    std::vector<int64_t> loadedIndices(bufferSize_);
    for (int64_t index = 0; index < size(); index++) {
        loadedIndices[index] = index;
    }
    loadedIndices_ = loadedIndices;
    stepCounter_ = (int64_t) size();
}

void C_Memory::clear() {
    /*!
     * Clears the data in C_Memory. This will **NOT** free the memory since it doesn't perform any memory de-allocation.
     * This is C++ backend of rlpack._C.memory.Memory.clear method.
     */
    statesCurrent_.clear();
    statesNext_.clear();
    rewards_.clear();
    actions_.clear();
    dones_.clear();
    priorities_.clear();
    probabilities_.clear();
    weights_.clear();
    if(sumTreeSharedPtr_ != nullptr){
        sumTreeSharedPtr_->reset();
    }
}

size_t C_Memory::size() {
    /*!
     * This method obtains the current size of C_Memory. This is the C++ backend of rlpack._C.memory.Memory.__len__
     * method, so length can be obtained by in-built python function len(memory).
     *
     * @return The size(or length) of C_Memory.
     */
    return dones_.size();
}

int64_t C_Memory::num_terminal_states() {
    /*!
     * Method to obtain the number of terminal states currently in C_Memory. This is the C++ backend of
     * rlpack._C.memory.Memory.num_terminal_states method.
     *
     * @return Number of terminal states so far.
     */
    return (int64_t) terminalStateIndices_.size();
}

int64_t C_Memory::tree_height() {
    /*!
     * Method to obtain the tree height of the sum tree if using a proportional prioritization strategy. This is
     * the C++ backend of rlpack._C.memory.Memory.tree_height. If not using proportional prioritization strategy,
     * calling this method will throw an error.
     *
     * @return The tree height of the tree built.
     */
     // sumTreeSharedPtr_ is set to nullptr by default and only changes when using proportional prioritization strategy.
    if(sumTreeSharedPtr_ == nullptr){
        throw std::runtime_error("Accessing `tree_height` method when not using proportional prioritization strategy");
    }
    return sumTreeSharedPtr_->get_tree_height();
}

torch::Tensor C_Memory::compute_probabilities(torch::Tensor &priorities, float_t alpha) {
    /*!
     * Method to compute probabilities when not using uniform prioritization strategy.
     *
     * @param priorities : The sampled priorities for which probabilities are to be computed.
     * @param alpha : The alpha value for prioritization. Refer C_Memory::sample for more information.
     * @return The tensor with probabilities corresponding to each priority.
     */
    auto prioritiesPowered = torch::pow(priorities, alpha);
    auto probabilities = prioritiesPowered / torch::sum(prioritiesPowered);
    return probabilities;
}

torch::Tensor C_Memory::compute_important_sampling_weights(torch::Tensor &probabilities,
                                                           int64_t currentSize,
                                                           float_t beta) {
    /*!
     * Method to compute the important sampling weights for each probabilities.
     *
     * @param probabilities : The input probabilities for which IS weights are to be computed.
     * @param currentSize : The current size of the C_Memory (see C_Memory::size)
     * @param beta : The beta value for prioritization. Refer C_Memory::sample for more information.
     * @return The tensor with important sampling weights corresponding to each probability.
     */
    auto weights = torch::pow(1 / (currentSize * probabilities), beta);
    auto maxWeightInBatch = weights.max().item<float_t>();
    weights = weights / maxWeightInBatch;
    return weights;
}

#pragma clang diagnostic pop
