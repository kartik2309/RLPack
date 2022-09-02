//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#include "C_Memory.h"

C_Memory::C_MemoryData::C_MemoryData(int64_t *bufferSizeRawPtr,
                                     int64_t *parallelismSizeThresholdRawPtr) {
  bufferSizeRawPtr_ = bufferSizeRawPtr;
  parallelismSizeThresholdRawPtr_ = parallelismSizeThresholdRawPtr;
  transitionInformationMap_["states_current"].reserve(*bufferSizeRawPtr_);
  transitionInformationMap_["states_next"].reserve(*bufferSizeRawPtr_);
  transitionInformationMap_["rewards"].reserve(*bufferSizeRawPtr_);
  transitionInformationMap_["actions"].reserve(*bufferSizeRawPtr_);
  transitionInformationMap_["dones"].reserve(*bufferSizeRawPtr_);
}

C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::vector<torch::Tensor>> C_Memory::C_MemoryData::deref_transition_information_map() {
  size_t size = current_size();
  std::vector<torch::Tensor> statesCurrent(size), statesNext(size), rewards(size), actions(size), dones(size);
  bool enableParallelism = size > *parallelismSizeThresholdRawPtr_;
  {
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(size) shared(transitionInformationMap_, statesCurrent, statesNext, rewards, actions, dones)
    for (int64_t index = 0; index < size; index++) {
      statesCurrent[index] = *transitionInformationMap_["states_current"][index];
      statesNext[index] = *transitionInformationMap_["states_next"][index];
      rewards[index] = *transitionInformationMap_["rewards"][index];
      actions[index] = *transitionInformationMap_["actions"][index];
      dones[index] = *transitionInformationMap_["dones"][index];
    }
  }
  std::map<std::string, std::vector<torch::Tensor>> _derefTransitionInformationMap = {
      {"states_current", statesCurrent},
      {"states_next", statesNext},
      {"rewards", rewards},
      {"actions", actions},
      {"dones", dones},
  };

  return _derefTransitionInformationMap;
}

std::map<std::string, std::vector<int64_t>> C_Memory::C_MemoryData::deref_terminal_state_indices() {
  size_t size = terminalStateIndexReferences_.size();
  std::vector<int64_t> terminalStateIndices(size);
  bool enableParallelism = size > *parallelismSizeThresholdRawPtr_;
  {
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(size) shared(terminalStateIndexReferences_, terminalStateIndices)
    for (int64_t index = 0; index < size; index++) {
      terminalStateIndices[index] = *terminalStateIndexReferences_[index];
    }
  }
  std::map<std::string, std::vector<int64_t>> _derefTerminalStates = {
      {"terminal_state_indices", terminalStateIndices},
  };

  return _derefTerminalStates;
}

void C_Memory::C_MemoryData::set_transition_information_reference(std::shared_ptr<torch::Tensor> &stateCurrent,
                                                                  std::shared_ptr<torch::Tensor> &stateNext,
                                                                  std::shared_ptr<torch::Tensor> &reward,
                                                                  std::shared_ptr<torch::Tensor> &action,
                                                                  std::shared_ptr<torch::Tensor> &done) {
  transitionInformationMap_["states_current"].push_back(stateCurrent);
  transitionInformationMap_["states_next"].push_back(stateNext);
  transitionInformationMap_["rewards"].push_back(reward);
  transitionInformationMap_["actions"].push_back(action);
  transitionInformationMap_["dones"].push_back(done);
}

void C_Memory::C_MemoryData::set_terminal_state_index_reference(const std::shared_ptr<int64_t> &terminalStateIndex) {
  terminalStateIndexReferences_.push_back(terminalStateIndex);
}

void C_Memory::C_MemoryData::initialize_data(std::map<std::string,
                                                      std::vector<torch::Tensor>> &transitionInformationMap,
                                             std::map<std::string, std::vector<int64_t>> &terminalStateIndicesMap) {
  auto terminalIndicesVector = terminalStateIndicesMap["terminal_state_indices"];
  std::vector<std::shared_ptr<int64_t>> terminalStateIndicesRef(terminalIndicesVector.size());
  auto terminalIndicesVectorSize = terminalIndicesVector.size();
  auto transitionsSize = transitionInformationMap["states_current"].size();
  std::vector<std::shared_ptr<torch::Tensor>> statesCurrent(transitionsSize), statesNext(transitionsSize),
      rewards(transitionsSize), actions(transitionsSize), dones(transitionsSize);
  bool enableParallelism = transitionsSize > *parallelismSizeThresholdRawPtr_;
  {
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(transitionsSize, transitionInformationMap) shared(statesCurrent, statesNext, rewards, actions, dones)
    for (int64_t index = 0; index < transitionsSize; index++) {
      statesCurrent[index] = std::make_shared<torch::Tensor>(transitionInformationMap["states_current"][index]);
      statesNext[index] = std::make_shared<torch::Tensor>(transitionInformationMap["states_next"][index]);
      rewards[index] = std::make_shared<torch::Tensor>(transitionInformationMap["actions"][index]);
      actions[index] = std::make_shared<torch::Tensor>(transitionInformationMap["rewards"][index]);
      dones[index] = std::make_shared<torch::Tensor>(transitionInformationMap["dones"][index]);
    }
  }
  {
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(terminalIndicesVectorSize, terminalIndicesVector) shared(terminalStateIndicesRef)
    for (int64_t index = 0; index < terminalIndicesVectorSize; index++) {
      terminalStateIndicesRef[index] = std::make_shared<int64_t>(terminalIndicesVector[index]);
    }
  }
  transitionInformationMap_ = {
      {"states_current", statesCurrent},
      {"states_next", statesNext},
      {"rewards", rewards},
      {"actions", actions},
      {"dones", dones},
  };
  terminalStateIndexReferences_ = terminalStateIndicesRef;
}

size_t C_Memory::C_MemoryData::current_size() {
  return transitionInformationMap_["dones"].size();
}

int64_t C_Memory::C_MemoryData::get_buffer_size() const {
  return *bufferSizeRawPtr_;
}

int64_t C_Memory::C_MemoryData::get_parallelism_size_threshold() const {
  return *parallelismSizeThresholdRawPtr_;
}

C_Memory::C_Memory(pybind11::int_ &bufferSize, pybind11::str &device, const pybind11::int_ &parallelismSizeThreshold) {
  bufferSize_ = bufferSize.cast<int64_t>();
  device_ = deviceMap[device.cast<std::string>()];
  parallelismSizeThreshold_ = parallelismSizeThreshold.cast<int64_t>();
  cMemoryData = std::make_shared<C_MemoryData>(&bufferSize_, &parallelismSizeThreshold_);
  loadedIndices_.reserve(bufferSize_);
}

C_Memory::C_Memory() {
  loadedIndices_.reserve(bufferSize_);
  cMemoryData = std::make_shared<C_MemoryData>(&bufferSize_, &parallelismSizeThreshold_);
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
  if (stepCounter_ < bufferSize_) {
    auto transitionInformation = std::make_shared<TransitionInformation_>();
    transitionInformation->stateCurrent = stateCurrent;
    transitionInformation->stateNext = stateNext;
    transitionInformation->reward = reward;
    transitionInformation->action = action;
    transitionInformation->done = done;
    transitionInformationBuffer_.push_back(transitionInformation);
    if (isTerminalState) {
      auto terminalStateIndex = std::make_shared<int64_t>((int64_t) size() - 1);
      terminalStateIndices_.push_back(*terminalStateIndex);
      cMemoryData->set_terminal_state_index_reference(terminalStateIndex);
    }
    if (size() < bufferSize_) {
      loadedIndices_.push_back(stepCounter_);
    }

    // Convert to shared pointer and add to transition information in C_MemoryData
    auto statesCurrentSharedPtr = std::make_shared<torch::Tensor>(transitionInformation->stateCurrent);
    auto statesNextSharedPtr = std::make_shared<torch::Tensor>(transitionInformation->stateNext);
    auto rewardsSharedPtr = std::make_shared<torch::Tensor>(transitionInformation->reward);
    auto actionsSharedPtr = std::make_shared<torch::Tensor>(transitionInformation->action);
    auto donesSharedPtr = std::make_shared<torch::Tensor>(transitionInformation->done);
    cMemoryData->set_transition_information_reference(statesCurrentSharedPtr,
                                                      statesNextSharedPtr,
                                                      rewardsSharedPtr,
                                                      actionsSharedPtr,
                                                      donesSharedPtr);
  } else {
    stepCounter_ = -1;
  }
  stepCounter_ += 1;
}

std::map<std::string, torch::Tensor> C_Memory::get_item(int64_t index) {
  if (index >= size()) {
    throw std::out_of_range("Given index exceeds the current_size of C_Memory");
  } else if (index < 0) {
    throw std::out_of_range("Index must be greater than 0");
  }
  auto item = transitionInformationBuffer_[index];
  std::map<std::string, torch::Tensor> returnItems = {
      {"states_current", item->stateCurrent},
      {"states_next", item->stateNext},
      {"reward", item->reward},
      {"action", item->action},
      {"done", item->done}
  };
  return returnItems;
}

void C_Memory::delete_item(int64_t index) {
  if (index != 0) {
    if (transitionInformationBuffer_[index]->done.flatten().item<int32_t>() == 1) {
      auto indexIter = std::find(terminalStateIndices_.begin(), terminalStateIndices_.end(), index);
      if (indexIter != terminalStateIndices_.end()) {
        terminalStateIndices_.erase(indexIter);
      } else {
        std::cerr << "Deletion of terminal state occurred but "
                     "terminal state was not found in `terminalStateIndices_`" << std::endl;
      }
    }
    transitionInformationBuffer_.erase(transitionInformationBuffer_.begin() + index);
  } else {
    if (transitionInformationBuffer_[0]->done.flatten().item<int32_t>() == 1) {
      terminalStateIndices_.pop_front();
    }
    transitionInformationBuffer_.pop_front();
  }
}

std::map<std::string, torch::Tensor> C_Memory::sample(int32_t batchSize,
                                                      float_t forceTerminalStateProbability) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float_t> distributionP(0, 1);
  std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                     (int64_t) terminalStateIndices_.size() - 1);
  std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
      sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize);
  auto loadedIndices = shuffle_loaded_indices(parallelismSizeThreshold_);
  std::vector<int64_t> loadedIndicesSlice = std::vector<int64_t>(loadedIndices.begin(),
                                                                 loadedIndices.begin() + batchSize);
  int64_t index = 0;
  bool forceTerminalState = false;
  float_t p = distributionP(generator);
  if (size() < batchSize) {
    throw std::out_of_range("Requested batch size is larger than current Memory current_size!");
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
  for (auto &loadedIndex : loadedIndicesSlice) {
    sampledStateCurrent[index] = transitionInformationBuffer_.at(loadedIndex)->stateCurrent;
    sampledStateNext[index] = transitionInformationBuffer_.at(loadedIndex)->stateNext;
    sampledRewards[index] = transitionInformationBuffer_.at(loadedIndex)->reward;
    sampledActions[index] = transitionInformationBuffer_.at(loadedIndex)->action;
    sampledDones[index] = transitionInformationBuffer_.at(loadedIndex)->done;
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
      {"states_next", statesNextStacked},
      {"rewards", rewardsStacked},
      {"actions", actionsStacked},
      {"dones", donesStacked}
  };
  return samples;
}

C_Memory::C_MemoryData C_Memory::view() const {
  return *cMemoryData;
}

void C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
  cMemoryData = std::make_shared<C_MemoryData>(viewC_MemoryData);
  auto transitionInformationDeref = cMemoryData->deref_transition_information_map();
  auto terminalIndicesDeref = cMemoryData->deref_terminal_state_indices();
  auto terminalIndicesDerefVector = terminalIndicesDeref["terminal_state_indices"];
  auto transitionsSize = cMemoryData->current_size();
  auto terminalIndicesDerefVectorSize = terminalIndicesDerefVector.size();
  std::deque<std::shared_ptr<TransitionInformation_>> transitionInformationBuffer(transitionsSize);
  std::deque<int64_t> terminalStateIndices(terminalIndicesDerefVectorSize);
  std::vector<int64_t> loadedIndices(transitionsSize);
  bool enableParallelism = transitionsSize > parallelismSizeThreshold_;
  {
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(transitionsSize, transitionInformationDeref) shared(transitionInformationBuffer, loadedIndices)
    for (int64_t index = 0; index < transitionsSize; index++) {
      auto transitionInformation = std::make_shared<TransitionInformation_>();
      transitionInformation->stateCurrent = transitionInformationDeref["states_current"][index];
      transitionInformation->stateNext = transitionInformationDeref["states_next"][index];
      transitionInformation->reward = transitionInformationDeref["rewards"][index];
      transitionInformation->action = transitionInformationDeref["actions"][index];
      transitionInformation->done = transitionInformationDeref["dones"][index];
      transitionInformationBuffer[index] = transitionInformation;
      loadedIndices[index] = index;
    }
  }
  {
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(terminalIndicesDerefVectorSize, terminalIndicesDerefVector) shared(terminalStateIndices)
    for (int64_t index = 0; index < terminalIndicesDerefVectorSize; index++) {
      terminalStateIndices[index] = terminalIndicesDerefVector[index];
    }
  }
  transitionInformationBuffer_ = transitionInformationBuffer;
  terminalStateIndices_ = terminalStateIndices;
  loadedIndices_ = loadedIndices;
}

void C_Memory::clear() {
  transitionInformationBuffer_.clear();
}

size_t C_Memory::size() {
  return transitionInformationBuffer_.size();
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
#pragma omp parallel for if(enableParallelism) default(none) firstprivate(loadedIndicesSize, distributionOfLoadedIndices, generator) shared(loadedIndices)
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

C_Memory::TransitionInformation_::TransitionInformation_() = default;

C_Memory::TransitionInformation_::~TransitionInformation_() = default;
