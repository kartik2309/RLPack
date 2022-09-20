//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
#include "C_Memory.h"

C_Memory::C_MemoryData::C_MemoryData() = default;

C_Memory::C_MemoryData::~C_MemoryData() = default;

std::map<std::string, std::deque<torch::Tensor>> C_Memory::C_MemoryData::dereferenceTransitionInformation() {
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

std::map<std::string, std::deque<int64_t>> C_Memory::C_MemoryData::dereferenceTerminalStateIndices() const {
  std::map<std::string, std::deque<int64_t>> dereferencedTerminalStates = {
      {"terminal_state_indices", *terminalIndicesReference_}
  };
  return dereferencedTerminalStates;
}

std::map<std::string, std::deque<float_t>> C_Memory::C_MemoryData::dereferencePriorities() const {
  std::map<std::string, std::deque<float_t>> dereferencedPriorities = {
      {"priorities", *prioritiesFloatReference_}
  };
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

C_Memory::C_Memory(pybind11::int_ &bufferSize, pybind11::str &device) {
  bufferSize_ = bufferSize.cast<int64_t>();
  device_ = deviceMap[device.cast<std::string>()];
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
  sumTreeSharedPtr_ = std::make_shared<SumTree_>(bufferSize_);
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
  sumTreeSharedPtr_ = std::make_shared<SumTree_>(bufferSize_);
}

C_Memory::~C_Memory() = default;

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
    delete_item(0, true);
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
  if (weight.item<float_t>() > maxWeight_) {
    maxWeight_ = weight.item<float_t>();
  }
  sumTreeSharedPtr_->insert(priority.item<float_t>());
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
  sumTreeSharedPtr_->update(index, priority.item<float_t>());
}

void C_Memory::delete_item(int64_t index, bool internal) {
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
        std::cerr << "Deletion of terminal state occurred but "
                     "terminal state was not found in `terminalStateIndices_`" << std::endl;
      }
    }
    dones_.erase(dones_.begin() + index);
    priorities_.erase(priorities_.begin() + index);
    probabilities_.erase(probabilities_.begin() + index);
    weights_.erase(weights_.begin() + index);
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
  }
  if (!internal){
    sumTreeSharedPtr_->update(index, 0);
  }
}

std::map<std::string, torch::Tensor> C_Memory::sample(int32_t batchSize,
                                                      float_t forceTerminalStateProbability,
                                                      int64_t parallelismSizeThreshold,
                                                      bool isPrioritized) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float_t> distributionP(0, 1);
  std::uniform_int_distribution<int64_t> distributionOfTerminalIndex(0,
                                                                     (int64_t) terminalStateIndices_.size() - 1);
  std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
      sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize),
      sampledPriorities(batchSize), sampledProbabilities(batchSize), sampledWeights(batchSize),
      sampledIndices(batchSize);
  std::vector<int64_t> loadedIndicesSlice;
  loadedIndicesSlice.reserve(batchSize);
  int64_t index = 0;
  bool forceTerminalState = false;
  if (!isPrioritized) {
    auto loadedIndices = get_loaded_indices(loadedIndices_, parallelismSizeThreshold);
    loadedIndicesSlice = std::vector<int64_t>(loadedIndices.begin(),
                                              loadedIndices.begin() + batchSize);
  } else {
    auto seedValues = get_priority_seeds(
        sumTreeSharedPtr_->get_cumulative_sum(), parallelismSizeThreshold);
    for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
      auto seedValue = seedValues[batchIndex];
      auto randomIndex = sumTreeSharedPtr_->sample(seedValue, (int64_t) size());
      loadedIndicesSlice.push_back(randomIndex);
    }
  }

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

  for (auto &loadedIndex : loadedIndicesSlice) {
    sampledStateCurrent[index] = statesCurrent_[loadedIndex];
    sampledStateNext[index] = statesNext_[loadedIndex];
    sampledRewards[index] = rewards_[loadedIndex];
    sampledActions[index] = actions_[loadedIndex];
    sampledDones[index] = dones_[loadedIndex];
    sampledPriorities[index] = priorities_[loadedIndex];
    sampledProbabilities[index] = probabilities_[loadedIndex];
    sampledWeights[index] = weights_[loadedIndex];
    sampledIndices[index] = torch::full({}, loadedIndex);
    index++;
  }
  auto statesCurrentStacked = torch::stack(sampledStateCurrent, 0).to(device_);
  auto statesNextStacked = torch::stack(sampledStateNext, 0).to(device_);

  auto targetDataType = statesNextStacked.dtype();
  auto targetSize = statesCurrentStacked.sizes();

  auto rewardsStacked = torch::stack(sampledRewards, 0).to(
      device_, targetDataType);

  auto actionsStacked = torch::stack(sampledActions, 0).to(
      device_, torch::kInt64);
  actionsStacked = adjust_dimensions(actionsStacked, targetSize);

  auto donesStacked = torch::stack(sampledDones, 0).to(
      device_, torch::kInt32);

  auto prioritiesStacked = torch::stack(sampledPriorities, 0).to(device_);
  auto probabilitiesStacked = torch::stack(sampledProbabilities, 0).to(device_);
  auto weightsStacked = torch::stack(sampledWeights, 0).to(
      device_, targetDataType);
  auto sampledIndicesStacked = torch::stack(sampledIndices, 0).to(device_);
  std::map<std::string, torch::Tensor> samples = {
      {"states_current", statesCurrentStacked},
      {"states_next", statesNextStacked},
      {"rewards", rewardsStacked},
      {"actions", actionsStacked},
      {"dones", donesStacked},
      {"priorities", prioritiesStacked},
      {"probabilities", probabilitiesStacked},
      {"weights", weightsStacked},
      {"random_indices", sampledIndicesStacked}
  };
  return samples;
}

void C_Memory::update_priorities(torch::Tensor &randomIndices,
                                 torch::Tensor &newPriorities,
                                 float_t alpha,
                                 float_t beta) {
  newPriorities = newPriorities.flatten();
  auto newProbabilities = compute_probabilities(newPriorities, alpha);
  auto newWeights = compute_important_sampling_weights(newProbabilities,
                                                       (int64_t) size(),
                                                       beta,
                                                       maxWeight_);
  randomIndices = randomIndices.flatten();
  auto size = randomIndices.size(0);
  for (int32_t index = 0; index < size; index++) {
    auto selectIndex = randomIndices[index].item<int64_t>();
    auto newPriority = newPriorities[index].item<float_t>();
    sumTreeSharedPtr_->update(selectIndex, newPriority);
    priorities_[selectIndex] = newPriorities[index];
    probabilities_[selectIndex] = newProbabilities[index];
    weights_[selectIndex] = newWeights[index];
    prioritiesFloat_[selectIndex] = newPriority;
  }
}

C_Memory::C_MemoryData C_Memory::view() const {
  return *cMemoryData;
}

void C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
  cMemoryData = std::make_shared<C_MemoryData>(viewC_MemoryData);
  auto transitionInformation = cMemoryData->dereferenceTransitionInformation();
  auto terminalStateIndices = cMemoryData->dereferenceTerminalStateIndices();
  auto prioritiesFloat = cMemoryData->dereferencePriorities();
  statesCurrent_ = transitionInformation["states_current"];
  statesNext_ = transitionInformation["states_next"];
  rewards_ = transitionInformation["rewards"];
  actions_ =transitionInformation["actions"];
  dones_ = transitionInformation["dones"];
  priorities_ = transitionInformation["priorities"];
  probabilities_ = transitionInformation["probabilities"];
  weights_ = transitionInformation["weights"];
  std::vector<int64_t> loadedIndices(bufferSize_);
  for (int64_t index = 0; index < size(); index++) {
    loadedIndices[index] = index;
  }
  std::copy(loadedIndices.begin(),
            loadedIndices.end(),
            loadedIndices_.begin());
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


std::vector<int64_t> C_Memory::get_loaded_indices(std::vector<int64_t> &loadedIndices,
                                                  int64_t parallelismSizeThreshold) {
  std::random_device rd;
  std::mt19937 generator(rd());
  auto loadedIndicesSize = (int64_t) loadedIndices.size();

  std::vector<int64_t> loadedIndicesCopy = std::vector<int64_t>(loadedIndices.begin(),
                                                                loadedIndices.end());
  std::uniform_int_distribution<int64_t> distributionOfLoadedIndices(0,
                                                                     loadedIndicesSize - 1);
  bool enableParallelism = loadedIndicesSize > parallelismSizeThreshold;

  // Parallel region for shuffling the `loadedIndices_`.
  {
#pragma omp parallel for if(enableParallelism) default(none) \
firstprivate(loadedIndicesSize, distributionOfLoadedIndices, generator) shared(loadedIndicesCopy)
    for (int64_t index = 0;
         index < loadedIndicesSize;
         index++) {
      int64_t randomLoadedIndex = distributionOfLoadedIndices(generator) % (loadedIndicesSize - index);
      std::iter_swap(loadedIndicesCopy.begin() + index, loadedIndicesCopy.begin() + index + randomLoadedIndex);
    }
  }
  // End of parallel region.
  return loadedIndices;
}

std::vector<float_t> C_Memory::get_priority_seeds(float_t cumulativeSum,
                                                  int64_t parallelismSizeThreshold) {
  std::random_device rd;
  std::mt19937 generator(rd());
  auto indexSize = (int64_t) cumulativeSum;
  std::uniform_real_distribution<float_t> distributionOfErrors(0.0, 1.0);
  std::uniform_int_distribution<int64_t> distributionOfIndices(0,
                                                               indexSize - 1);
  std::vector<float_t> seeds(indexSize);
  bool enableParallelism = parallelismSizeThreshold < indexSize;
  {
#pragma omp parallel for if(enableParallelism) default(none)\
firstprivate(indexSize, distributionOfErrors, generator)\
shared(seeds)
    for (int64_t index = 0; index < indexSize; index++) {
      auto randomError = distributionOfErrors(generator);
      seeds[index] = (float_t) index + randomError;
    }
  }
  {
#pragma omp parallel for if(enableParallelism) default(none)\
firstprivate(indexSize, distributionOfIndices, generator)\
shared(seeds)
    for (int64_t index = 0; index < indexSize; index++) {
      int64_t randomLoadedIndex = distributionOfIndices(generator) % (indexSize - index);
      std::iter_swap(seeds.begin() + index, seeds.begin() + index + randomLoadedIndex);
    }
  }
  return seeds;
}

torch::Tensor C_Memory::compute_probabilities(torch::Tensor &priorities, float_t alpha) {
  auto prioritiesPowered = torch::pow(priorities, alpha);
  auto probabilities = prioritiesPowered / torch::sum(prioritiesPowered);
  return probabilities;
}

torch::Tensor C_Memory::compute_important_sampling_weights(torch::Tensor &probabilities,
                                                           int64_t currentSize,
                                                           float_t beta,
                                                           float_t maxWeight) {
  auto weights = torch::pow(currentSize * probabilities, -beta);
  weights = weights / maxWeight;
  return weights;
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

C_Memory::SumTreeNode_::SumTreeNode_(SumTreeNode_ *parent,
                                     float_t value,
                                     int64_t treeIndex,
                                     int64_t index,
                                     int64_t treeLevel,
                                     bool allowTraversal,
                                     SumTreeNode_ *leftNode,
                                     SumTreeNode_ *rightNode) {
  parent_ = parent;
  leftNode_ = leftNode;
  rightNode_ = rightNode;
  treeIndex_ = treeIndex;
  index_ = index;
  treeLevel_ = treeLevel;
  allowTraversal_ = allowTraversal;
  value_ = value;

  if (leftNode_ != nullptr || rightNode_ != nullptr) {
    isLeaf_ = false;
  }
}

C_Memory::SumTreeNode_::~SumTreeNode_() = default;

void C_Memory::SumTreeNode_::set_value(float_t newValue) {
  value_ = newValue;
}

[[maybe_unused]] void C_Memory::SumTreeNode_::remove_left_node() {
  leftNode_ = nullptr;
}

[[maybe_unused]] void C_Memory::SumTreeNode_::remove_right_node() {
  rightNode_ = nullptr;
}

void C_Memory::SumTreeNode_::set_left_node(C_Memory::SumTreeNode_ *node) {
  leftNode_ = node;
  if (leftNode_ != nullptr || rightNode_ != nullptr) {
    isLeaf_ = false;
  }
}

void C_Memory::SumTreeNode_::set_right_node(C_Memory::SumTreeNode_ *node) {
  if (leftNode_ == nullptr) {
    throw std::runtime_error("Tried to add right node before setting left node!");
  }
  rightNode_ = node;
  if (leftNode_ != nullptr || rightNode_ != nullptr) {
    isLeaf_ = false;
  }
}

float_t C_Memory::SumTreeNode_::get_value() const {
  return value_;
}

int64_t C_Memory::SumTreeNode_::get_tree_index() const {
  return treeIndex_;
}

int64_t C_Memory::SumTreeNode_::get_index() const {
  return index_;
}

int64_t C_Memory::SumTreeNode_::get_tree_level() const {
  return treeLevel_;
}

C_Memory::SumTreeNode_ *C_Memory::SumTreeNode_::get_parent() {
  return parent_;
}

C_Memory::SumTreeNode_ *C_Memory::SumTreeNode_::get_left_node() {
  return leftNode_;
}

C_Memory::SumTreeNode_ *C_Memory::SumTreeNode_::get_right_node() {
  return rightNode_;
}

bool C_Memory::SumTreeNode_::is_leaf() const {
  return isLeaf_;
}

bool C_Memory::SumTreeNode_::is_traversal_allowed() const {
  return allowTraversal_;
}

bool C_Memory::SumTreeNode_::is_head() {
  if (parent_ != nullptr) {
    return false;
  }
  return true;
}

void C_Memory::SumTreeNode_::set_parent_node(C_Memory::SumTreeNode_ *parent) {
  parent_ = parent;
}

void C_Memory::SumTreeNode_::set_allow_traversal(bool allowTraversal) {
  allowTraversal_ = allowTraversal;
}

C_Memory::SumTree_::SumTree_(int32_t bufferSize) {
  bufferSize_ = bufferSize;
  leaves_.reserve(bufferSize_);
  sumTree_.reserve(bufferSize_);
  std::vector<float_t> priorities(bufferSize, 0.0);
  auto nullOptional = std::make_optional<std::vector<SumTreeNode_ * >>();
  nullOptional = std::nullopt;
  create_tree(priorities, nullOptional);
}

C_Memory::SumTree_::~SumTree_() = default;

C_Memory::SumTree_::SumTree_() = default;

void C_Memory::SumTree_::create_tree(std::vector<float_t> &priorities,
                                     std::optional<std::vector<SumTreeNode_ *>> &children) {
  if (children.has_value()) {
    assert(priorities.size() == children.value().size());
    treeHeight_++;
  }
  std::vector<float_t> prioritiesForTree(priorities.begin(), priorities.end());
  std::vector<float_t> prioritiesSum;
  std::vector<SumTreeNode_ *> childrenForRecursion;
  if (prioritiesForTree.size() % 2 != 0 && prioritiesForTree.size() != 1) {
    prioritiesForTree.push_back(0);
    if (children.has_value()) {
      children.value().push_back(nullptr);
    }
  }
  prioritiesSum.reserve((prioritiesForTree.size() / 2) + 1);
  childrenForRecursion.reserve((prioritiesForTree.size() / 2) + 1);
  for (int64_t index = 0; index < prioritiesForTree.size(); index = index + 2) {
    auto leftPriority = prioritiesForTree[index];
    auto rightPriority = prioritiesForTree[index + 1];
    auto sum = leftPriority + rightPriority;
    prioritiesSum.push_back(sum);
    if (!children.has_value()) {
      auto parent = new SumTreeNode_(nullptr, sum, (int64_t) sumTree_.size() + 2);
      auto leftNode = new SumTreeNode_(parent, leftPriority, (int64_t) sumTree_.size(), index);
      auto rightNode = new SumTreeNode_(parent, rightPriority, (int64_t) sumTree_.size() + 1, index + 1);
      parent->set_left_node(leftNode);
      parent->set_right_node(rightNode);
      leftNode->set_allow_traversal(true);
      rightNode->set_allow_traversal(true);
      sumTree_.push_back(leftNode);
      sumTree_.push_back(rightNode);
      sumTree_.push_back(parent);
      leaves_[index] = leftNode;
      leaves_[index + 1] = rightNode;
      childrenForRecursion.push_back(parent);
    } else {
      auto leftChild = children.value()[index];
      auto rightChild = children.value()[index + 1];
      auto parent = new SumTreeNode_(nullptr,
                                     sum,
                                     (int64_t) sumTree_.size(),
                                     -1,
                                     treeHeight_,
                                     false,
                                     leftChild,
                                     rightChild);
      leftChild->set_parent_node(parent);
      if (rightChild != nullptr) {
        rightChild->set_parent_node(parent);
      }
      sumTree_.push_back(parent);
      childrenForRecursion.push_back(parent);
    }
  }
  if (prioritiesSum.size() == 1) {
    return;
  }
  children = childrenForRecursion;
  create_tree(prioritiesSum, children);
}

void C_Memory::SumTree_::reset() {
  sumTree_.clear();
  leaves_.clear();
}

void C_Memory::SumTree_::insert(float_t value) {
  if (stepCounter_ >= bufferSize_) {
    stepCounter_ = 0;
  }
  update(stepCounter_, value);
  stepCounter_++;
}

int64_t C_Memory::SumTree_::sample(float_t seedValue, int64_t currentSize) {
  auto parent = sumTree_.back();
  auto node = traverse(parent, seedValue);
  auto index = node->get_index();
  if (index > currentSize) {
    std::cerr << "Larger index than current size was generated " << index << std::endl;
    index = index % currentSize;
  }
  return index;
}

void C_Memory::SumTree_::update(int64_t index, float_t value) {
  auto leaf = leaves_[index];
  auto change = value - leaf->get_value();
  leaf->set_value(value);
  propagate_changes_upwards(leaf->get_parent(), change);
}

float_t C_Memory::SumTree_::get_cumulative_sum() {
  auto parentNode = sumTree_.back();
  return parentNode->get_value();
}

int64_t C_Memory::SumTree_::get_tree_height() {
  auto parent = sumTree_.back();
  return parent->get_tree_level();
}

void C_Memory::SumTree_::propagate_changes_upwards(SumTreeNode_ *node,
                                                   float_t change) {
  auto newValue = node->get_value() + change;
  node->set_value(newValue);
  sumTree_[node->get_tree_index()]->set_value(newValue);
  node->set_allow_traversal(true);
  sumTree_[node->get_tree_index()]->set_allow_traversal(true);
  if (!node->is_head()) {
    node = node->get_parent();
    propagate_changes_upwards(node, change);
  }
}

C_Memory::SumTreeNode_ *C_Memory::SumTree_::traverse(C_Memory::SumTreeNode_ *node, float_t value) {
  if (!node->is_traversal_allowed()){
    std::cerr << "WARNING: Traversed through restricted Node" << std::endl;
  }
  if (!node->is_leaf()) {
    auto leftNode = node->get_left_node();
    auto rightNode = node->get_right_node();
    if (leftNode->get_value() >= value || rightNode == nullptr) {
      node = leftNode;
    } else {
      value = value - leftNode->get_value();
      node = rightNode;
    }
    node = traverse(node, value);
  }
  return node;
}

#pragma clang diagnostic pop