//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#include "C_Memory.h"

C_Memory::C_MemoryData::C_MemoryData(int64_t *bufferSizeRawPtr,
                                     int64_t *parallelismSizeThresholdRawPtr) {
  bufferSizeRawPtr_ = bufferSizeRawPtr;
  parallelismSizeThresholdRawPtr_ = parallelismSizeThresholdRawPtr;
}

C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::vector<torch::Tensor>>
C_Memory::C_MemoryData::deref_transition_information_map() {
  size_t size = _size();
  std::vector<torch::Tensor> statesCurrent(size), statesNext(size),
      rewards(size), actions(size), dones(size);
  bool enableParallelism = size > *parallelismSizeThresholdRawPtr_;
  {
#pragma omp parallel for if(enableParallelism)  \
default(none) firstprivate(size)  \
shared(transitionInformationMap_, statesCurrent, statesNext, rewards, actions, dones)
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

std::map<std::string, std::vector<int64_t>>
C_Memory::C_MemoryData::deref_terminal_state_indices() {
  size_t size = terminalStateIndexReferences_.size();
  std::vector<int64_t> terminalStateIndices(size);
  bool enableParallelism = size > *parallelismSizeThresholdRawPtr_;
  {
#pragma omp parallel for if(enableParallelism) \
default(none) firstprivate(size) \
shared(terminalStateIndexReferences_, terminalStateIndices)
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
                                                                  std::shared_ptr<torch::Tensor> &done,
                                                                  std::shared_ptr<torch::Tensor> &priority,
                                                                  std::shared_ptr<torch::Tensor> &probability,
                                                                  std::shared_ptr<torch::Tensor> &weight) {
  transitionInformationMap_["states_current"].push_back(stateCurrent);
  transitionInformationMap_["states_next"].push_back(stateNext);
  transitionInformationMap_["rewards"].push_back(reward);
  transitionInformationMap_["actions"].push_back(action);
  transitionInformationMap_["dones"].push_back(done);
  transitionInformationMap_["priorities"].push_back(priority);
  transitionInformationMap_["probabilities"].push_back(probability);
  transitionInformationMap_["weights"].push_back(weight);
}

void
C_Memory::C_MemoryData::set_transition_information_reference(int64_t index,
                                                             std::shared_ptr<torch::Tensor> &stateCurrent,
                                                             std::shared_ptr<torch::Tensor> &stateNext,
                                                             std::shared_ptr<torch::Tensor> &reward,
                                                             std::shared_ptr<torch::Tensor> &action,
                                                             std::shared_ptr<torch::Tensor> &done,
                                                             std::shared_ptr<torch::Tensor> &priority,
                                                             std::shared_ptr<torch::Tensor> &probability,
                                                             std::shared_ptr<torch::Tensor> &weight) {
  transitionInformationMap_["states_current"][index] = stateCurrent;
  transitionInformationMap_["states_next"][index] = stateNext;
  transitionInformationMap_["rewards"][index] = reward;
  transitionInformationMap_["actions"][index] = action;
  transitionInformationMap_["dones"][index] = done;
  transitionInformationMap_["priorities"][index] = priority;
  transitionInformationMap_["probabilities"][index] = probability;
  transitionInformationMap_["weights"][index] = weight;
}

void
C_Memory::C_MemoryData::set_terminal_state_index_reference(const std::shared_ptr<int64_t> &terminalStateIndex) {
  terminalStateIndexReferences_.push_back(terminalStateIndex);
}

void
C_Memory::C_MemoryData::set_terminal_state_index_reference(int64_t index,
                                                           const std::shared_ptr<int64_t> &terminalStateIndex) {
  terminalStateIndexReferences_[index] = terminalStateIndex;
}

void
C_Memory::C_MemoryData::delete_transition_information_reference(int64_t index) {
  if (index != 0) {
    transitionInformationMap_["states_current"].erase(
        transitionInformationMap_["states_current"].begin() + index);
    transitionInformationMap_["states_next"].erase(
        transitionInformationMap_["states_next"].begin() + index);
    transitionInformationMap_["rewards"].erase(
        transitionInformationMap_["rewards"].begin() + index);
    transitionInformationMap_["actions"].erase(
        transitionInformationMap_["actions"].begin() + index);
    transitionInformationMap_["dones"].erase(
        transitionInformationMap_["dones"].begin() + index);
    transitionInformationMap_["priorities"].erase(
        transitionInformationMap_["priorities"].begin() + index);
    transitionInformationMap_["probabilities"].erase(
        transitionInformationMap_["probabilities"].begin() + index);
    transitionInformationMap_["weights"].erase(
        transitionInformationMap_["weights"].begin() + index);
  } else {
    transitionInformationMap_["states_current"].pop_front();
    transitionInformationMap_["states_next"].pop_front();
    transitionInformationMap_["rewards"].pop_front();
    transitionInformationMap_["actions"].pop_front();
    transitionInformationMap_["dones"].pop_front();
    transitionInformationMap_["priorities"].pop_front();
    transitionInformationMap_["probabilities"].pop_front();
    transitionInformationMap_["weights"].pop_front();
  }
}

void
C_Memory::C_MemoryData::delete_terminal_state_index_reference(int64_t terminalStateIndex) {
  if (terminalStateIndex != 0) {
    terminalStateIndexReferences_.erase(terminalStateIndexReferences_.begin() + terminalStateIndex);
  } else {
    terminalStateIndexReferences_.pop_front();
  }
}

void C_Memory::C_MemoryData::update_transition_priority_references(int64_t index,
                                                                   std::shared_ptr<torch::Tensor> &priority,
                                                                   std::shared_ptr<torch::Tensor> &probability,
                                                                   std::shared_ptr<torch::Tensor> &weight) {
  transitionInformationMap_["priorities"][index] = priority;
  transitionInformationMap_["probabilities"][index] = probability;
  transitionInformationMap_["weights"][index] = weight;
}


void
C_Memory::C_MemoryData::initialize_data(
    std::map<std::string, std::vector<torch::Tensor>> &transitionInformationMap,
    std::map<std::string, std::vector<int64_t>> &terminalStateIndicesMap) {
  auto terminalIndicesVector = terminalStateIndicesMap["terminal_state_indices"];
  std::deque<std::shared_ptr<int64_t>> terminalStateIndicesRef(terminalIndicesVector.size());
  auto terminalIndicesVectorSize = terminalIndicesVector.size();
  auto transitionsSize = transitionInformationMap["states_current"].size();
  std::deque<std::shared_ptr<torch::Tensor>> statesCurrent(transitionsSize), statesNext(transitionsSize),
      rewards(transitionsSize), actions(transitionsSize), dones(transitionsSize);
  bool enableParallelism = transitionsSize > *parallelismSizeThresholdRawPtr_;
  {
#pragma omp parallel for if(enableParallelism) \
default(none) firstprivate(transitionsSize, transitionInformationMap) \
shared(statesCurrent, statesNext, rewards, actions, dones)
    for (int64_t index = 0; index < transitionsSize; index++) {
      statesCurrent[index] = std::make_shared<torch::Tensor>(transitionInformationMap["states_current"][index]);
      statesNext[index] = std::make_shared<torch::Tensor>(transitionInformationMap["states_next"][index]);
      rewards[index] = std::make_shared<torch::Tensor>(transitionInformationMap["actions"][index]);
      actions[index] = std::make_shared<torch::Tensor>(transitionInformationMap["rewards"][index]);
      dones[index] = std::make_shared<torch::Tensor>(transitionInformationMap["dones"][index]);
    }
  }
  {
#pragma omp parallel for if(enableParallelism) \
default(none) firstprivate(terminalIndicesVectorSize, terminalIndicesVector) \
shared(terminalStateIndicesRef)
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

size_t
C_Memory::C_MemoryData::_size() {
  return transitionInformationMap_["dones"].size();
}

int64_t C_Memory::C_MemoryData::_num_of_terminal_states() {
  return (int64_t) terminalStateIndexReferences_.size();
}

int64_t
C_Memory::C_MemoryData::get_buffer_size() const {
  return *bufferSizeRawPtr_;
}

int64_t
C_Memory::C_MemoryData::get_parallelism_size_threshold() const {
  return *parallelismSizeThresholdRawPtr_;
}

C_Memory::C_Memory(pybind11::int_ &bufferSize,
                   pybind11::str &device,
                   const pybind11::int_ &parallelismSizeThreshold) {
  bufferSize_ = bufferSize.cast<int64_t>();
  device_ = deviceMap[device.cast<std::string>()];
  parallelismSizeThreshold_ = parallelismSizeThreshold.cast<int64_t>();
  cMemoryData = std::make_shared<C_MemoryData>(
      &bufferSize_, &parallelismSizeThreshold_);
  loadedIndices_.reserve(bufferSize_);
  maxWeight_ = torch::zeros({}, torch::kFloat32);
}

C_Memory::C_Memory() {
  loadedIndices_.reserve(bufferSize_);
  cMemoryData = std::make_shared<C_MemoryData>(
      &bufferSize_, &parallelismSizeThreshold_);
  maxWeight_ = torch::zeros({}, torch::kFloat32);
}

C_Memory::~C_Memory() = default;

void
C_Memory::insert(torch::Tensor &stateCurrent,
                 torch::Tensor &stateNext,
                 torch::Tensor &reward,
                 torch::Tensor &action,
                 torch::Tensor &done,
                 torch::Tensor &priority,
                 torch::Tensor &probability,
                 torch::Tensor &weight) {
  if (size() > bufferSize_) {
    delete_item(0);
  }
  if (stepCounter_ >= bufferSize_) {
    stepCounter_ = -1;
  }
  bool isTerminalState = done.item<float_t>() == 1;
  // Initialize TransitionInformation_ object and add its shared_ptr to transitionInformationBuffer_
  auto transitionInformation = std::make_shared<TransitionInformation_>();
  transitionInformation->stateCurrent = stateCurrent;
  transitionInformation->stateNext = stateNext;
  transitionInformation->reward = reward;
  transitionInformation->action = action;
  transitionInformation->done = done;
  transitionInformation->priority = priority;
  transitionInformation->probability = probability;
  transitionInformation->weight = weight;
  transitionInformationBuffer_.push_back(transitionInformation);
  if (isTerminalState) {
    auto terminalStateIndex = std::make_shared<int64_t>((int64_t) size() - 1);
    terminalStateIndices_.push_back(*terminalStateIndex);
    cMemoryData->set_terminal_state_index_reference(terminalStateIndex);
    terminalStateToTransitionBufferMap_[numOfTerminalStates() - 1] = transitionInformation;
    transitionBufferToTerminalStateMap_[transitionInformation] = numOfTerminalStates() - 1;
  }
  if (size() < bufferSize_) {
    loadedIndices_.push_back(stepCounter_);
  }
  // Push transition priority value to priorities_ buffer.
  priorities_.push_back(transitionInformation->priority.item<float_t>());
  // Update maxWeight_ if required.
  if (maxWeight_.item<float_t>() < transitionInformation->weight.item<float_t>()) {
    maxWeight_ = transitionInformation->weight;
  }
  // Get the shared pointer reference to tensors from transitions.
  auto statesCurrentSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->stateCurrent);
  auto statesNextSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->stateNext);
  auto rewardsSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->reward);
  auto actionsSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->action);
  auto donesSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->done);
  auto prioritySharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->priority);
  auto probabilitySharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->probability);
  auto weightSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->weight);
  cMemoryData->set_transition_information_reference(statesCurrentSharedPtr,
                                                    statesNextSharedPtr,
                                                    rewardsSharedPtr,
                                                    actionsSharedPtr,
                                                    donesSharedPtr,
                                                    prioritySharedPtr,
                                                    probabilitySharedPtr,
                                                    weightSharedPtr);
  stepCounter_ += 1;
}

std::map<std::string, torch::Tensor>
C_Memory::get_item(int64_t index) {
  if (index >= size()) {
    throw std::out_of_range("Given index exceeds the _size of C_Memory");
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

void
C_Memory::delete_item(int64_t index) {
  if (index > size()) {
    throw std::out_of_range("Passed index was larger than memory size!");
  }
  auto transition = transitionInformationBuffer_[index];
  if (index != 0) {
    if (transition->done.flatten().item<int32_t>() != 1) {
      transitionInformationBuffer_.erase(transitionInformationBuffer_.begin() + index);
    } else {
      auto terminalStateIndex = transitionBufferToTerminalStateMap_[transition];
      terminalStateIndices_.erase(terminalStateIndices_.begin() + terminalStateIndex);
    }
    priorities_.erase(priorities_.begin() + index);
  } else {
    if (transition->done.flatten().item<int32_t>() != 1) {
      transitionInformationBuffer_.pop_front();
    } else {
      auto terminalStateIndex = transitionBufferToTerminalStateMap_[transition];
      if (terminalStateIndex != 0) {
        std::cerr << "Expected head transition to be mapped with head terminal state index buffer!" << std::endl;
      } else {
        transitionBufferToTerminalStateMap_.erase(transition);
        terminalStateToTransitionBufferMap_.erase(terminalStateIndex);
        cMemoryData->delete_transition_information_reference(index);
        cMemoryData->delete_terminal_state_index_reference(terminalStateIndex);
        terminalStateIndices_.pop_front();
      }
    }
    priorities_.pop_front();
  }
}

void
C_Memory::set_item(int64_t index,
                   torch::Tensor &stateCurrent,
                   torch::Tensor &stateNext,
                   torch::Tensor &reward,
                   torch::Tensor &action,
                   torch::Tensor &done,
                   torch::Tensor &priority,
                   torch::Tensor &probability,
                   torch::Tensor &weight) {
  if (index >= size()) {
    throw std::out_of_range("Passed index is greater than current size!");
  }
  if (size() > bufferSize_) {
    delete_item(0);
  }
  bool isTerminalState = done.item<float_t>() == 1;
  auto transitionInformation = std::make_shared<TransitionInformation_>();
  transitionInformation->stateCurrent = stateCurrent;
  transitionInformation->stateNext = stateNext;
  transitionInformation->reward = reward;
  transitionInformation->action = action;
  transitionInformation->done = done;
  transitionInformation->priority = priority;
  transitionInformation->probability = probability;
  transitionInformation->weight = weight;
  transitionInformationBuffer_[index] = transitionInformation;
  if (isTerminalState) {
    if (terminalStateToTransitionBufferMap_.count(index) < 1) {
      auto terminalStateIndex = std::make_shared<int64_t>(index);
      terminalStateIndices_.push_back(index);
      cMemoryData->set_terminal_state_index_reference(index, terminalStateIndex);
      terminalStateToTransitionBufferMap_[numOfTerminalStates() - 1] = transitionInformation;
      transitionBufferToTerminalStateMap_[transitionInformation] = numOfTerminalStates() - 1;
    }
  }
  loadedIndices_[index] = index;
  priorities_[index] = transitionInformation->priority.item<float_t>();

  // Convert to shared pointer and add to transition information in C_MemoryData
  auto statesCurrentSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->stateCurrent);
  auto statesNextSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->stateNext);
  auto rewardsSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->reward);
  auto actionsSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->action);
  auto donesSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->done);
  auto prioritySharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->priority);
  auto probabilitySharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->probability);
  auto weightSharedPtr = std::make_shared<torch::Tensor>(
      transitionInformation->weight);
  cMemoryData->set_transition_information_reference(index,
                                                    statesCurrentSharedPtr,
                                                    statesNextSharedPtr,
                                                    rewardsSharedPtr,
                                                    actionsSharedPtr,
                                                    donesSharedPtr,
                                                    prioritySharedPtr,
                                                    probabilitySharedPtr,
                                                    weightSharedPtr);
}

std::map<std::string, torch::Tensor>
C_Memory::sample(int32_t batchSize,
                 float_t forceTerminalStateProbability) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float_t> distributionP(0, 1);
  std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                     numOfTerminalStates());
  std::uniform_int_distribution<int64_t> distribution(0, (int64_t) size());
  std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
      sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize);
  std::vector<int64_t> loadedIndicesSlice(batchSize);
  int64_t index = 0;
  bool forceTerminalState = false;

  /*
   * Create a list of random indices with uniform distribution without replacement.
   * These indices are then used to samples from transitionInformationBuffer_.
   */
  for (int32_t index_lis = 0; index_lis != batchSize; index_lis++) {
    auto randomIndex = distribution(generator);
    if (
        std::find(loadedIndicesSlice.begin(),
                  loadedIndicesSlice.end(),
                  randomIndex) != loadedIndicesSlice.end()) {
      continue;
    }
    loadedIndicesSlice[index_lis] = randomIndex;
  }
  float_t p = distributionP(generator);
  if (size() < batchSize) {
    throw std::out_of_range("Requested batch size is larger than current Memory _size!");
  }
  if (p < forceTerminalStateProbability && numOfTerminalStates() > 1) {
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

std::map<std::string, torch::Tensor> C_Memory::sample(int32_t batchSize,
                                                      float_t forceTerminalStateProbability,
                                                      float_t alpha) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float_t> distributionP(0, 1);
  std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                     numOfTerminalStates());
  std::discrete_distribution<int64_t> distribution(priorities_.begin(), priorities_.end());
  std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
      sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize),
      randomIndices(batchSize);
  std::vector<int64_t> loadedIndicesSlice(batchSize);
  int64_t index = 0;
  bool forceTerminalState = false;
  float_t p = distributionP(generator);
  if (size() < batchSize) {
    throw std::out_of_range("Requested batch size is larger than current Memory _size!");
  }
  if (p < forceTerminalStateProbability && numOfTerminalStates() > 1) {
    forceTerminalState = true;
  }
  /*
   * Create a list of random indices with discrete distribution (with weights/priorities) without replacement.
   * These indices are then used to samples from transitionInformationBuffer_.
   */
  for (int32_t index_lis = 0; index_lis != batchSize; index_lis++) {
    auto randomIndex = distribution(generator);
    if (
        std::find(loadedIndicesSlice.begin(),
                  loadedIndicesSlice.end(),
                  randomIndex) != loadedIndicesSlice.end()) {
      continue;
    }
    loadedIndicesSlice[index_lis] = randomIndex;
  }
  if (forceTerminalState) {
    int64_t randomIndexToInsertTerminalState = distributionOfTerminalIndex(generator) % batchSize;
    int64_t randomTerminalStateInfoIndex = distributionOfTerminalIndex(generator);
    int64_t randomTerminalStateIndex = terminalStateIndices_[randomTerminalStateInfoIndex];
    loadedIndicesSlice[randomIndexToInsertTerminalState] = randomTerminalStateIndex;
  }
  for (auto &randomIndex : loadedIndicesSlice) {
    sampledStateCurrent[index] = transitionInformationBuffer_.at(randomIndex)->stateCurrent;
    sampledStateNext[index] = transitionInformationBuffer_.at(randomIndex)->stateNext;
    sampledRewards[index] = transitionInformationBuffer_.at(randomIndex)->reward;
    sampledActions[index] = transitionInformationBuffer_.at(randomIndex)->action;
    sampledDones[index] = transitionInformationBuffer_.at(randomIndex)->done;
    randomIndices[index] = torch::full({}, randomIndex);
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
  auto randomIndicesStacked = torch::stack(randomIndices).to(device_, torch::kInt64);

  std::map<std::string, torch::Tensor> samples = {
      {"states_current", statesCurrentStacked},
      {"states_next", statesNextStacked},
      {"rewards", rewardsStacked},
      {"actions", actionsStacked},
      {"dones", donesStacked},
      {"random_index", randomIndicesStacked}
  };
  return samples;
}

void C_Memory::update_transition_priorities(const std::vector<int64_t> &indices,
                                            const std::vector<torch::Tensor> &newPriorities,
                                            float_t beta) {
  int32_t baseIndex = 0;
  auto newPrioritiesStacked = torch::stack(newPriorities, 0);
  auto newProbabilities = compute_probabilities(newPrioritiesStacked);
  auto newWeights = compute_important_sampling_weights((int64_t) size(),
                                                       newProbabilities,
                                                       maxWeight_,
                                                       beta);
  for (auto &index : indices) {
    transitionInformationBuffer_.at(index)->priority = newPriorities[baseIndex];
    transitionInformationBuffer_.at(index)->probability = newProbabilities[baseIndex];
    transitionInformationBuffer_.at(index)->weight = newWeights[baseIndex];

    // Update the shared pointer references and point them to new values.
    auto prioritySharedPtr = std::make_shared<torch::Tensor>(
        transitionInformationBuffer_.at(index)->priority);
    auto probabilitySharedPtr = std::make_shared<torch::Tensor>(
        transitionInformationBuffer_.at(index)->probability);
    auto weightSharedPtr = std::make_shared<torch::Tensor>(
        transitionInformationBuffer_.at(index)->weight);
    cMemoryData->update_transition_priority_references(index,
                                                       prioritySharedPtr,
                                                       probabilitySharedPtr,
                                                       weightSharedPtr);
    baseIndex++;
  }
}

C_Memory::C_MemoryData
C_Memory::view() const {
  return *cMemoryData;
}

void
C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
  cMemoryData = std::make_shared<C_MemoryData>(viewC_MemoryData);
  auto transitionInformationDeref = cMemoryData->deref_transition_information_map();
  auto terminalIndicesDeref = cMemoryData->deref_terminal_state_indices();
  auto terminalIndicesDerefVector = terminalIndicesDeref["terminal_state_indices"];
  auto transitionsSize = cMemoryData->_size();
  auto terminalIndicesDerefVectorSize = cMemoryData->_num_of_terminal_states();
  std::deque<std::shared_ptr<TransitionInformation_>> transitionInformationBuffer(transitionsSize);
  std::deque<int64_t> terminalStateIndices(terminalIndicesDerefVectorSize);
  std::vector<int64_t> loadedIndices(transitionsSize);
  std::map<int64_t, std::shared_ptr<TransitionInformation_>> terminalStateToTransitionBufferMap;
  std::map<std::shared_ptr<TransitionInformation_>, int64_t> transitionBufferToTerminalStateMap;
  bool enableParallelism = transitionsSize > parallelismSizeThreshold_;
  {
#pragma omp parallel for if(enableParallelism) \
default(none) firstprivate(transitionsSize, transitionInformationDeref) \
shared(transitionInformationBuffer, loadedIndices)
    for (int64_t index = 0; index < transitionsSize; index++) {
      auto transitionInformation = std::make_shared<TransitionInformation_>();
      transitionInformation->stateCurrent = transitionInformationDeref["states_current"][index];
      transitionInformation->stateNext = transitionInformationDeref["states_next"][index];
      transitionInformation->reward = transitionInformationDeref["rewards"][index];
      transitionInformation->action = transitionInformationDeref["actions"][index];
      transitionInformation->done = transitionInformationDeref["dones"][index];
      transitionInformationBuffer.at(index) = transitionInformation;
      loadedIndices.at(index) = index;
    }
  }
  {
#pragma omp parallel for if(enableParallelism) \
default(none) firstprivate(terminalIndicesDerefVectorSize, terminalIndicesDerefVector) \
shared(terminalStateIndices, transitionInformationBuffer, terminalStateToTransitionBufferMap, \
transitionBufferToTerminalStateMap)
    for (int64_t index = 0; index < terminalIndicesDerefVectorSize; index++) {
      terminalStateIndices[index] = terminalIndicesDerefVector[index];
      {
#pragma omp critical
        terminalStateToTransitionBufferMap.emplace(
            index, transitionInformationBuffer[terminalIndicesDerefVector[index]]);
        transitionBufferToTerminalStateMap.emplace(
            transitionInformationBuffer[terminalIndicesDerefVector[index]], index);
      }
    }
  }
  transitionInformationBuffer_ = transitionInformationBuffer;
  terminalStateIndices_ = terminalStateIndices;
  loadedIndices_ = loadedIndices;
  terminalStateToTransitionBufferMap_ = terminalStateToTransitionBufferMap;
  transitionBufferToTerminalStateMap_ = transitionBufferToTerminalStateMap;
}

void
C_Memory::clear() {
  transitionInformationBuffer_.clear();
}

size_t
C_Memory::size() {
  return transitionInformationBuffer_.size();
}

int64_t
C_Memory::numOfTerminalStates() {
  return (int64_t) terminalStateIndices_.size();
}

torch::Tensor
C_Memory::adjust_dimensions(torch::Tensor &tensor, c10::IntArrayRef &targetDimensions) {
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

torch::Tensor C_Memory::compute_probabilities(torch::Tensor &priorities) {
  return priorities / torch::sum(priorities);
}

torch::Tensor C_Memory::compute_important_sampling_weights(int64_t currentSize,
                                                           const torch::Tensor &probability,
                                                           const torch::Tensor &maxWeight,
                                                           float_t beta) {
  auto important_sampling_weights = torch::pow(currentSize * probability, -beta) / maxWeight;
  return important_sampling_weights;
}

C_Memory::TransitionInformation_::TransitionInformation_() = default;

C_Memory::TransitionInformation_::~TransitionInformation_() = default;
