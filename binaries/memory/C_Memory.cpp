//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#include "C_Memory.h"

C_Memory::C_MemoryData::C_MemoryData() = default;

C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::deque<torch::Tensor>> C_Memory::C_MemoryData::derefCoreData() {
    std::map<std::string, std::deque<torch::Tensor>> derefData = {
            {"states_current", *coreDataPtr["states_current"]},
            {"states_next",    *coreDataPtr["states_next"]},
            {"rewards",        *coreDataPtr["rewards"]},
            {"actions",        *coreDataPtr["actions"]},
            {"dones",          *coreDataPtr["dones"]},
    };

    return derefData;
}

std::map<std::string, std::deque<int64_t>> C_Memory::C_MemoryData::derefTerminalStateIndices() const {
    std::map<std::string, std::deque<int64_t>> derefTerminalStates = {
            {"terminal_state_indices", *terminalStatesIndicesPtr}
    };
    return derefTerminalStates;
}

C_Memory::C_Memory(pybind11::int_ &bufferSize, pybind11::str &device) {
    bufferSize_ = bufferSize.cast<int64_t>();
    device_ = deviceMap[device.cast<std::string>()];
    cMemoryData.coreDataPtr["states_current"] = &statesCurrent_;
    cMemoryData.coreDataPtr["states_next"] = &statesNext_;
    cMemoryData.coreDataPtr["rewards"] = &rewards_;
    cMemoryData.coreDataPtr["actions"] = &actions_;
    cMemoryData.coreDataPtr["dones"] = &dones_;
    cMemoryData.terminalStatesIndicesPtr = &terminalStateIndices_;
    loadedIndices_.reserve(bufferSize_);
}

C_Memory::C_Memory() {
    cMemoryData.coreDataPtr["states_current"] = &statesCurrent_;
    cMemoryData.coreDataPtr["states_next"] = &statesNext_;
    cMemoryData.coreDataPtr["rewards"] = &rewards_;
    cMemoryData.coreDataPtr["actions"] = &actions_;
    cMemoryData.coreDataPtr["dones"] = &dones_;
    cMemoryData.terminalStatesIndicesPtr = &terminalStateIndices_;
    loadedIndices_.reserve(bufferSize_);
}

C_Memory::~C_Memory() = default;

void C_Memory::insert(torch::Tensor &stateCurrent,
                      torch::Tensor &stateNext,
                      torch::Tensor &reward,
                      torch::Tensor &action,
                      torch::Tensor &done,
                      bool isTerminalState) {
    if (size() > bufferSize_) {
        delete_item(0);
    }
    statesCurrent_.push_back(stateCurrent);
    statesNext_.push_back(stateNext);
    rewards_.push_back(reward);
    actions_.push_back(action);
    dones_.push_back(done);

    if (isTerminalState) {
        terminalStateIndices_.push_back((int64_t) size() - 1);
    }
    if (size() < bufferSize_) {
        loadedIndices_.push_back(step_counter_);
        step_counter_ += 1;
    }
}

std::map<std::string, torch::Tensor> C_Memory::get_item(int64_t index) {
    std::map<std::string, torch::Tensor> returnItems = {
            {"states_current", statesCurrent_[index]},
            {"states_next",    statesNext_[index]},
            {"rewards",        rewards_[index]},
            {"actions",        actions_[index]},
            {"dones",          dones_[index]}
    };
    return returnItems;
}

void C_Memory::delete_item(int64_t index) {
    if (index != 0) {
        statesCurrent_.erase(statesCurrent_.begin() + index);
        statesNext_.erase(statesNext_.begin() + index);
        rewards_.erase(rewards_.begin() + index);
        actions_.erase(actions_.begin() + index);
        if (dones_[index].flatten().item<int32_t>() == 1) {
            auto indexIter = std::find(terminalStateIndices_.begin(), terminalStateIndices_.end(), index);
            if (indexIter != terminalStateIndices_.end()) {
                terminalStateIndices_.erase(indexIter);
            } else {
                std::cerr << "Deletion of terminal state occurred but "
                             "terminal state was not found in `terminalStateIndices_`" << std::endl;
            }
        }
        dones_.erase(dones_.begin() + index);
    } else {
        statesCurrent_.pop_front();
        statesNext_.pop_front();
        rewards_.pop_front();
        actions_.pop_front();
        dones_.pop_front();
        if (dones_[0].flatten().item<int32_t>() == 1) {
            terminalStateIndices_.pop_front();
        }
    }
}

std::map<std::string, torch::Tensor> C_Memory::sample(int32_t batchSize,
                                                      float_t forceTerminalStateProbability,
                                                      int64_t parallelismSizeThreshold) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float_t> distributionP(0, 1);
    std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                       (int64_t) terminalStateIndices_.size() - 1);
    std::uniform_int_distribution<int64_t> distributionOfLoadedIndices(0,
                                                                       (int64_t) loadedIndices_.size() - 1);

    std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
            sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize);

    auto loadedIndices = shuffle_loaded_indices(parallelismSizeThreshold);
    std::vector<int64_t> loadedIndicesSlice = std::vector<int64_t>(loadedIndices.begin(),
                                                                   loadedIndices.begin() + batchSize);
    int64_t index = 0;
    bool forceTerminalState = false;
    float_t p = distributionP(generator);
    if (size() < batchSize) {
        throw std::out_of_range("Requested batch size is larger than current Memory size!");
    }
    if (p < forceTerminalStateProbability && terminalStateIndices_.size() > 1) {
        forceTerminalState = true;
    }
    if (forceTerminalState) {
        int64_t randomIndexToInsertTerminalState = distributionOfTerminalIndex(generator) % batchSize;
        int64_t randomTerminalStateInfoIndex = distributionOfTerminalIndex(generator);
        int64_t randomTerminalStateIndex = terminalStateIndices_[randomTerminalStateInfoIndex];
        loadedIndicesSlice[randomIndexToInsertTerminalState] = randomTerminalStateIndex;
    }
    for (auto &loadedIndex: loadedIndicesSlice) {
        sampledStateCurrent[index] = statesCurrent_[loadedIndex];
        sampledStateNext[index] = statesNext_[loadedIndex];
        sampledRewards[index] = rewards_[loadedIndex];
        sampledActions[index] = actions_[loadedIndex];
        sampledDones[index] = dones_[loadedIndex];
        index++;
    }
    auto statesCurrentStacked = torch::stack(sampledStateCurrent, 0).to(device_);
    auto statesNextStacked = torch::stack(sampledStateNext, 0).to(device_);

    auto targetDataType = statesNextStacked.dtype();
    auto targetSize = statesCurrentStacked.sizes();

    auto rewardsStacked = torch::stack(sampledRewards, 0).to(
            device_, targetDataType);
    rewardsStacked = adjust_dimensions(rewardsStacked, targetSize);

    auto actionsStacked = torch::stack(sampledActions, 0).to(
            device_, torch::kInt64);
    actionsStacked = adjust_dimensions(actionsStacked, targetSize);

    auto donesStacked = torch::stack(sampledDones, 0).to(
            device_, torch::kInt32);
    donesStacked = adjust_dimensions(donesStacked, targetSize);

    std::map<std::string, torch::Tensor> samples = {
            {"states_current", statesCurrentStacked},
            {"states_next",    statesNextStacked},
            {"rewards",        rewardsStacked},
            {"actions",        actionsStacked},
            {"dones",          donesStacked}
    };
    return samples;
}

C_Memory::C_MemoryData C_Memory::view() {
    C_MemoryData viewC_MemoryData;
    viewC_MemoryData.coreDataPtr["states_current"] = &statesCurrent_;
    viewC_MemoryData.coreDataPtr["states_next"] = &statesNext_;
    viewC_MemoryData.coreDataPtr["rewards"] = &rewards_;
    viewC_MemoryData.coreDataPtr["actions"] = &actions_;
    viewC_MemoryData.coreDataPtr["dones"] = &dones_;
    viewC_MemoryData.terminalStatesIndicesPtr = &terminalStateIndices_;
    return viewC_MemoryData;
}

void C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
    statesCurrent_ = *viewC_MemoryData.coreDataPtr["states_current"];
    statesNext_ = *viewC_MemoryData.coreDataPtr["states_next"];
    rewards_ = *viewC_MemoryData.coreDataPtr["rewards"];
    actions_ = *viewC_MemoryData.coreDataPtr["actions"];
    dones_ = *viewC_MemoryData.coreDataPtr["dones"];
    terminalStateIndices_ = *viewC_MemoryData.terminalStatesIndicesPtr;
}

void C_Memory::clear() {
    statesCurrent_.clear();
    statesNext_.clear();
    rewards_.clear();
    actions_.clear();
    dones_.clear();
}

size_t C_Memory::size() {
    assert(statesCurrent_.size() == statesNext_.size() == rewards_.size() == actions_.size() == dones_.size());
    return dones_.size();
}

std::vector<int64_t> C_Memory::shuffle_loaded_indices(int64_t parallelismSizeThreshold) {
    std::random_device rd;
    std::mt19937 generator(rd());
    auto loadedIndicesSize = (int64_t) loadedIndices_.size();

    std::vector<int64_t> loadedIndices = std::vector<int64_t>(loadedIndices_.begin(),
                                                              loadedIndices_.end());
    std::uniform_int_distribution<int64_t> distributionOfLoadedIndices(0,
                                                                       loadedIndicesSize - 1);
    bool enableParallelism = loadedIndicesSize > parallelismSizeThreshold;

    // Parallel region for shuffling the `loadedIndices_`.
    {
#pragma omp parallel for if(enableParallelism) default(none) private(loadedIndicesSize, distributionOfLoadedIndices, generator) shared(loadedIndices)
        for (int64_t index = 0;
             index < loadedIndicesSize;
             index++) {
            int64_t randomLoadedIndex = distributionOfLoadedIndices(generator) % (loadedIndicesSize - index);
            std::iter_swap(loadedIndices.begin() + index, loadedIndices.begin() + index + randomLoadedIndex);
        }
    }
    // End of parallel region.

    return loadedIndices;
}

torch::Tensor C_Memory::adjust_dimensions(torch::Tensor &tensor, c10::IntArrayRef &targetDimensions) {
    auto currentSize = tensor.sizes();
    auto diffSizes = (int32_t) targetDimensions.size() - (int32_t) currentSize.size();
    if (diffSizes > 0) {
        for (int32_t _ = 0; _ != diffSizes; _++) {
            tensor = tensor.unsqueeze(-1);
        }
    } else {
        for (int32_t _ = 0; _ != diffSizes; _++) {
            tensor = tensor.squeeze(-1);
        }
    }
    return tensor;
}
