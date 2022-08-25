//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#include "C_Memory.h"

C_Memory::C_MemoryData::C_MemoryData() = default;

C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::vector<torch::Tensor>> C_Memory::C_MemoryData::derefCoreData() {
    std::map<std::string, std::vector<torch::Tensor>> derefData = {
            {"states_current", *coreDataPtr["states_current"]},
            {"states_next",    *coreDataPtr["states_next"]},
            {"rewards",        *coreDataPtr["rewards"]},
            {"actions",        *coreDataPtr["actions"]},
            {"dones",          *coreDataPtr["dones"]},
    };

    return derefData;
}

std::vector<int64_t> C_Memory::C_MemoryData::derefTerminalStateIndices() const {
    std::vector<int64_t> derefTerminalStates = *terminalStatesIndicesPtr;
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

void C_Memory::reserve(int64_t bufferSize) {
    bufferSize_ = bufferSize;
    statesCurrent_.reserve(bufferSize_);
    statesNext_.reserve(bufferSize_);
    rewards_.reserve(bufferSize_);
    actions_.reserve(bufferSize_);
    dones_.reserve(bufferSize_);
}

std::vector<torch::Tensor> C_Memory::get_item(int64_t index) {
    std::vector<torch::Tensor> returnItems = {
            statesCurrent_[index],
            statesNext_[index],
            rewards_[index],
            actions_[index],
            dones_[index],
    };
    return returnItems;
}

void C_Memory::delete_item(int64_t index) {
    statesCurrent_.erase(statesCurrent_.begin() + index);
    statesNext_.erase(statesNext_.begin() + index);
    rewards_.erase(rewards_.begin() + index);
    actions_.erase(actions_.begin() + index);
    dones_.erase(dones_.begin() + index);
}


std::map<std::string, torch::Tensor> C_Memory::sample(int32_t batchSize,
                                                      float_t forceTerminalStateProbability) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float_t> distributionP(0, 1);
    std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                       (int64_t) terminalStateIndices_.size() - 1);
    std::uniform_int_distribution<int64_t> distributionOfLoadedIndices(0,
                                                                       (int64_t) loadedIndices_.size() - 1);

    std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
            sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize);

    auto loadedIndices = shuffle_loaded_indices();
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
    std::map<std::string, torch::Tensor> samples = {{"states_current", torch::stack(sampledStateCurrent, 0).to(
            device_)},
                                                    {"states_next",    torch::stack(sampledStateNext, 0).to(device_)},
                                                    {"rewards",        torch::stack(sampledRewards, 0).to(device_)},
                                                    {"actions",        torch::stack(sampledActions, 0).to(device_)},
                                                    {"dones",          torch::stack(sampledDones, 0).to(device_)}};
    return samples;
}

std::map<std::string, torch::Tensor> C_Memory::sample(int32_t batchSize) {
    std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
            sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize);
    auto loadedIndices = shuffle_loaded_indices();
    std::vector<int64_t> loadedIndicesSlice = std::vector<int64_t>(loadedIndices.begin(),
                                                                   loadedIndices.begin() + batchSize);
    int64_t index = 0;
    if ((int32_t) size() < batchSize) {
        throw std::out_of_range("Requested batch size is larger than current Memory size!");
    }
    for (auto &loadedIndex: loadedIndicesSlice) {
        sampledStateCurrent[index] = statesCurrent_[loadedIndex];
        sampledStateNext[index] = statesNext_[loadedIndex];
        sampledRewards[index] = rewards_[loadedIndex];
        sampledActions[index] = actions_[loadedIndex];
        sampledDones[index] = dones_[loadedIndex];
        index++;
    }
    std::map<std::string, torch::Tensor> samples = {{"states_current", torch::stack(sampledStateCurrent, 0).to(
            device_)},
                                                    {"states_next",    torch::stack(sampledStateNext, 0).to(device_)},
                                                    {"rewards",        torch::stack(sampledRewards, 0).to(device_)},
                                                    {"actions",        torch::stack(sampledActions, 0).to(device_)},
                                                    {"dones",          torch::stack(sampledDones, 0).to(device_)}};
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

std::vector<int64_t> C_Memory::shuffle_loaded_indices() {
    std::random_device rd;
    std::mt19937 generator(rd());
    auto loadedIndicesSize = (int64_t) loadedIndices_.size();

    std::vector<int64_t> loadedIndices = std::vector<int64_t>(loadedIndices_.begin(),
                                                              loadedIndices_.end());
    std::uniform_int_distribution<int64_t> distributionOfLoadedIndices(0,
                                                                       loadedIndicesSize - 1);

    // Parallel region for shuffling the `loadedIndices_`.
    {
#pragma omp parallel for default(none) shared(loadedIndices, loadedIndicesSize, distributionOfLoadedIndices, generator)
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
