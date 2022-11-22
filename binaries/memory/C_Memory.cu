//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
#include "C_Memory.cuh"

C_Memory::C_MemoryData::C_MemoryData() = default;

C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::deque<torch::Tensor>> C_Memory::C_MemoryData::dereference_transition_information() {
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
    std::map<std::string, std::deque<int64_t>> dereferencedTerminalStates = {
            {"terminal_state_indices", *terminalIndicesReference_}};
    return dereferencedTerminalStates;
}

std::map<std::string, std::deque<float_t>> C_Memory::C_MemoryData::dereference_priorities() const {
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
    transitionInformationReference_[key] = reference;
}

void C_Memory::C_MemoryData::set_terminal_state_indices_reference(
        std::deque<int64_t> *&terminalStateIndicesReference) {
    terminalIndicesReference_ = terminalStateIndicesReference;
}

void C_Memory::C_MemoryData::set_priorities_reference(
        std::deque<float_t> *&prioritiesFloatReference) {
    prioritiesFloatReference_ = prioritiesFloatReference;
}

C_Memory::C_Memory(const pybind11::int_ &bufferSize,
                   const pybind11::str &device,
                   const pybind11::int_ &prioritizationStrategyCode,
                   const pybind11::int_ &batchSize) {
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
                loadedIndicesSliceToShuffle_[batchIndex] = randomIndex;
            }
//            offloadInt64_->reset();
//            offloadInt64_->shuffle(loadedIndicesSliceToShuffle_, parallelismSizeThreshold);
            memcpy(&loadedIndicesSlice_[0], &offloadInt64_->result[0], sizeof(int64_t) * batchSize_);
            break;
        }
        case 2: {
            // Rank-Based prioritization sampling.
            int64_t previousQuantileIndex = 0, generatedRandomIndex = 0;
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
                loadedIndicesSliceToShuffle_[index] = loadedIndices_[generatedRandomIndex];
                previousQuantileIndex = segmentQuantileIndex;
                index++;
            }
            index = 0;
//            offloadInt64_->reset();
//            offloadInt64_->shuffle(loadedIndicesSliceToShuffle_, parallelismSizeThreshold);
            memcpy(&loadedIndicesSlice_[0], &offloadInt64_->result[0], sizeof(int64_t) * batchSize_);
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
                                 torch::Tensor &newPriorities,
                                 torch::Tensor &newProbabilities,
                                 torch::Tensor &newWeights) {
    if (prioritizationStrategyCode_ == 0) {
        throw std::runtime_error("`update_priorities` method called in C++ backend when C_Memory is un-prioritized!");
    }
    newPriorities = newPriorities.flatten();
    randomIndices = randomIndices.flatten();
    auto size = randomIndices.size(0);
    for (int32_t index = 0; index < size; index++) {
        auto selectIndex = randomIndices[index].item<int64_t>();
        priorities_[selectIndex] = newPriorities[index];
        probabilities_[selectIndex] = newProbabilities[index];
        weights_[selectIndex] = newWeights[index];
        prioritiesFloat_[selectIndex] = newPriorities[index].item<float_t>();
    }
}

C_Memory::C_MemoryData C_Memory::view() const {
    return *cMemoryData;
}

void C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
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
    statesCurrent_.clear();
    statesNext_.clear();
    rewards_.clear();
    actions_.clear();
    dones_.clear();
    priorities_.clear();
    probabilities_.clear();
    weights_.clear();
    sumTreeSharedPtr_->reset();
}

size_t C_Memory::size() {
    return dones_.size();
}

int64_t C_Memory::num_terminal_states() {
    return (int64_t) terminalStateIndices_.size();
}

int64_t C_Memory::tree_height() {
    return sumTreeSharedPtr_->get_tree_height();
}

torch::Tensor C_Memory::compute_probabilities(torch::Tensor &priorities, float_t alpha) {
    auto prioritiesPowered = torch::pow(priorities, alpha);
    auto probabilities = prioritiesPowered / torch::sum(prioritiesPowered);
    return probabilities;
}

torch::Tensor C_Memory::compute_important_sampling_weights(torch::Tensor &probabilities,
                                                           int64_t currentSize,
                                                           float_t beta) {
    auto weights = torch::pow(1 / (currentSize * probabilities), beta);
    auto maxWeightInBatch = weights.max().item<float_t>();
    weights = weights / maxWeightInBatch;
    return weights;
}

#pragma clang diagnostic pop
