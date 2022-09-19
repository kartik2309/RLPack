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
  if (isTerminalState) {
    terminalStateIndices_.push_back((int64_t) size() - 1);
  }
  prioritiesFloat_.push_back(priority.item<float_t>());
  if (size() < bufferSize_) {
    loadedIndices_.push_back(stepCounter_);
    stepCounter_ += 1;
  }
  sumTreeSharedPtr_->insert(priority.item<float_t>());
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
  std::uniform_int_distribution<int64_t> distributionOfLoadedIndices(0,
                                                                     (int64_t) loadedIndices_.size() - 1);
  std::vector<torch::Tensor> sampledStateCurrent(batchSize), sampledStateNext(batchSize),
      sampledRewards(batchSize), sampledActions(batchSize), sampledDones(batchSize),
      sampledPriorities(batchSize), sampledProbabilities(batchSize), sampledWeights(batchSize),
      sampledIndices(batchSize);
  std::vector<int64_t> loadedIndicesSlice;
  loadedIndicesSlice.reserve(batchSize);
  int64_t index = 0;
  bool forceTerminalState = false;

  if (!isPrioritized) {
    auto loadedIndices = shuffle_loaded_indices(parallelismSizeThreshold);
    loadedIndicesSlice = std::vector<int64_t>(loadedIndices.begin(),
                                              loadedIndices.begin() + batchSize);
  } else {
    auto loadedPriorities = prioritiesFloat_;
    std::discrete_distribution<int64_t> discreteDistribution(loadedPriorities.begin(), loadedPriorities.end());
    for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
      int64_t randomIndex = discreteDistribution(generator);
      loadedPriorities[randomIndex] = 0;
      loadedIndicesSlice.push_back(randomIndex);
      discreteDistribution.reset();
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
      device_, targetDataType).unsqueeze(-1);
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

C_Memory::C_MemoryData C_Memory::view() const {
  return *cMemoryData;
}

void C_Memory::initialize(C_Memory::C_MemoryData &viewC_MemoryData) {
  cMemoryData = std::make_shared<C_MemoryData>(viewC_MemoryData);
  auto transitionInformation = cMemoryData->dereferenceTransitionInformation();
  auto terminalStateIndices = cMemoryData->dereferenceTerminalStateIndices();
  auto prioritiesFloat = cMemoryData->dereferencePriorities();

  std::copy(transitionInformation["states_current"].begin(),
            transitionInformation["states_current"].end(),
            statesCurrent_.begin());
  std::copy(transitionInformation["states_next"].begin(),
            transitionInformation["states_next"].end(),
            statesNext_.begin());
  std::copy(transitionInformation["rewards"].begin(),
            transitionInformation["rewards"].end(),
            rewards_.begin());
  std::copy(transitionInformation["actions"].begin(),
            transitionInformation["actions"].end(),
            actions_.begin());
  std::copy(transitionInformation["dones"].begin(),
            transitionInformation["dones"].end(),
            dones_.begin());
  std::copy(transitionInformation["priorities"].begin(),
            transitionInformation["priorities"].end(),
            priorities_.begin());
  std::copy(transitionInformation["probabilities"].begin(),
            transitionInformation["probabilities"].end(),
            probabilities_.begin());
  std::copy(transitionInformation["weights"].begin(),
            transitionInformation["weights"].end(),
            weights_.begin());
  std::copy(terminalStateIndices["terminal_state_indices"].begin(),
            terminalStateIndices["terminal_state_indices"].end(),
            terminalStateIndices_.begin());
  std::copy(prioritiesFloat["priorities"].begin(),
            prioritiesFloat["priorities"].end(),
            prioritiesFloat_.begin());

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
}

size_t C_Memory::size() {
  assert(
      statesCurrent_.size()
          == statesNext_.size()
          == rewards_.size()
          == actions_.size()
          == dones_.size()
          == priorities_.size()
          == probabilities_.size()
          == weights_.size());
  return dones_.size();
}

int64_t C_Memory::num_terminal_states() {
  return (int64_t) terminalStateIndices_.size();
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
#pragma omp parallel for if(enableParallelism) default(none) \
firstprivate(loadedIndicesSize, distributionOfLoadedIndices, generator) shared(loadedIndices)
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

C_Memory::SumTreeNode_::SumTreeNode_(SumTreeNode_ *parent,
                                     float_t value,
                                     int64_t index,
                                     SumTreeNode_ *leftNode,
                                     SumTreeNode_ *rightNode) {
  parent_ = parent;
  leftNode_ = leftNode;
  rightNode_ = rightNode;
  index_ = index;
  value_ = value;

  if (leftNode_ != nullptr || rightNode_ != nullptr) {
    isLeaf_ = false;
  }
}

C_Memory::SumTreeNode_::~SumTreeNode_() = default;

void C_Memory::SumTreeNode_::change_value(float_t newValue) {
  value_ = newValue;
}

void C_Memory::SumTreeNode_::remove_child_node(int8_t code) {
  switch (code) {
    case 0:leftNode_ = nullptr;
    case 1:rightNode_ = nullptr;
    default:std::cerr << "Invalid code was received, did not remove any node!" << std::endl;
      break;
  }
  if (leftNode_ == nullptr && rightNode_ == nullptr) {
    isLeaf_ = true;
  }
}

float_t C_Memory::SumTreeNode_::get_value() const {
  return value_;
}

int64_t C_Memory::SumTreeNode_::get_index() const {
  return index_;
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

bool C_Memory::SumTreeNode_::is_head() {
  if (parent_ != nullptr) {
    return false;
  }
  return true;
}

void C_Memory::SumTreeNode_::set_child_node(int8_t code, SumTreeNode_ *node) {
  if (leftNode_ == nullptr && code == 1) {
    throw std::runtime_error("Tried to insert right node before setting left node!");
  }
  switch (code) {
    case 0:leftNode_ = node;
    case 1:rightNode_ = node;
    default:std::cerr << "Invalid code was received, did not remove any node!" << std::endl;
      break;
  }
  if (leftNode_ != nullptr || rightNode_ != nullptr) {
    isLeaf_ = false;
  }
}
void C_Memory::SumTreeNode_::set_parent_node(C_Memory::SumTreeNode_ *parent) {
  parent_ = parent;
}

C_Memory::SumTree_::SumTree_(int32_t bufferSize) {
  bufferSize_ = bufferSize;
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
  prioritiesSum.reserve(prioritiesForTree.size() / 2);
  childrenForRecursion.reserve(prioritiesForTree.size() / 2);
  for (int64_t index = 0; index < prioritiesForTree.size(); index = index + 2) {
    auto leftPriority = prioritiesForTree[index];
    auto rightPriority = prioritiesForTree[index + 1];
    auto sum = leftPriority + rightPriority;
    prioritiesSum.push_back(sum);
    if (!children.has_value()) {
      auto parent = new SumTreeNode_(nullptr, sum, -1);
      auto leftNode = new SumTreeNode_(parent, leftPriority, index);
      auto rightNode = new SumTreeNode_(parent, rightPriority, index);
      parent->set_child_node(0, leftNode);
      parent->set_child_node(1, rightNode);
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
                                     -1,
                                     leftChild,
                                     rightChild);
      leftChild->set_parent_node(parent);
      rightChild->set_parent_node(parent);
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
  auto node = leaves_[stepCounter_];
  auto change = node->get_value() - value;
  node->change_value(value);
  propagate_changes_upwards(node, change);
  stepCounter_++;
}

int64_t C_Memory::SumTree_::sample_index(std::uniform_real_distribution<float_t> *distribution,
                                         std::mt19937 *generator) {
  auto distributionDereferenced = *distribution;
  auto generatorDereferenced = *generator;
  auto value = distributionDereferenced(generatorDereferenced);

  auto parent = sumTree_.back();
  return traverse(parent, value);
}

void C_Memory::SumTree_::update(std::vector<int64_t> &indices,
                                std::vector<float_t> &values) {
  assert(indices.size() == values.size());
  for (int32_t index = 0; index < indices.size(); index++) {
    update(index, values[index]);
  }
}

void C_Memory::SumTree_::update(int64_t index, float_t value) {
  auto leaf = leaves_[index];
  auto change = leaf->get_value() - value;
  leaf->change_value(value);
  propagate_changes_upwards(leaf, change);
}

float_t C_Memory::SumTree_::get_cumulative_sum() {
  return sumTree_.back()->get_value();
}

void C_Memory::SumTree_::propagate_changes_upwards(SumTreeNode_ *node,
                                                   float_t change) {
  while (!node->is_head()) {
    node->change_value(node->get_value() + change);
    node = node->get_parent();
    propagate_changes_upwards(node, change);
  }
}

int64_t C_Memory::SumTree_::traverse(C_Memory::SumTreeNode_ *node, float_t value) {
  if (!node->is_leaf()) {
    auto leftNode = node->get_left_node();
    auto rightNode = node->get_right_node();
    if (leftNode->get_value() >= value) {
      node = leftNode;
    } else {
      value = value - rightNode->get_value();
      node = rightNode;
    }
    traverse(node, value);
  }
  return node->get_index();
}

#pragma clang diagnostic pop