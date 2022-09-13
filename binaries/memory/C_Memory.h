//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#ifndef RLPACK_C_MEMORY_H
#define RLPACK_C_MEMORY_H

#include <random>
#include <omp.h>
#include <vector>
#include <deque>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

class C_Memory {
 public:

  struct C_MemoryData {
    C_MemoryData(int64_t *bufferSizeRawPtr,
                 int64_t *parallelismSizeThresholdRawPtr);
    ~C_MemoryData();

    std::map<std::string, std::vector<torch::Tensor>> deref_transition_information_map();
    std::map<std::string, std::vector<int64_t>> deref_terminal_state_indices();
    std::map<std::string, std::vector<float_t>> deref_priorities();
    void set_transition_information_reference(std::shared_ptr<torch::Tensor> &stateCurrent,
                                              std::shared_ptr<torch::Tensor> &stateNext,
                                              std::shared_ptr<torch::Tensor> &reward,
                                              std::shared_ptr<torch::Tensor> &action,
                                              std::shared_ptr<torch::Tensor> &done,
                                              std::shared_ptr<torch::Tensor> &priority,
                                              std::shared_ptr<torch::Tensor> &probability,
                                              std::shared_ptr<torch::Tensor> &weight);
    void set_transition_information_reference(int64_t index,
                                              std::shared_ptr<torch::Tensor> &stateCurrent,
                                              std::shared_ptr<torch::Tensor> &stateNext,
                                              std::shared_ptr<torch::Tensor> &reward,
                                              std::shared_ptr<torch::Tensor> &action,
                                              std::shared_ptr<torch::Tensor> &done,
                                              std::shared_ptr<torch::Tensor> &priority,
                                              std::shared_ptr<torch::Tensor> &probability,
                                              std::shared_ptr<torch::Tensor> &weight);
    void set_terminal_state_index_reference(const std::shared_ptr<int64_t> &terminalStateIndex);
    void set_terminal_state_index_reference(int64_t index,
                                            const std::shared_ptr<int64_t> &terminalStateIndex);
    void set_priority_reference(const std::shared_ptr<float_t> &priority);
    void set_priority_reference(int64_t index,
                                const std::shared_ptr<float_t> &priority);
    void delete_transition_information_reference(int64_t index);
    void delete_terminal_state_index_reference(int64_t terminalStateIndex);
    void delete_priority_reference(int64_t index);
    void update_transition_priority_references(int64_t index,
                                               std::shared_ptr<torch::Tensor> &priority,
                                               std::shared_ptr<torch::Tensor> &probability,
                                               std::shared_ptr<torch::Tensor> &weight);
    void initialize_data(std::map<std::string, std::vector<torch::Tensor>> &transitionInformationMap,
                         std::map<std::string, std::vector<int64_t>> &terminalStateIndicesMap,
                         std::map<std::string, std::vector<float_t>> &prioritiesReferencesMap);
    size_t _size();
    int64_t _num_of_terminal_states();
    [[nodiscard]] int64_t get_buffer_size() const;
    [[nodiscard]] int64_t get_parallelism_size_threshold() const;
   private:
    std::map<std::string, std::deque<std::shared_ptr<torch::Tensor>>> transitionInformationMap_;
    std::deque<std::shared_ptr<int64_t>> terminalStateIndexReferences_;
    std::deque<std::shared_ptr<float_t>> prioritiesReferences_;
    int64_t *bufferSizeRawPtr_;
    int64_t *parallelismSizeThresholdRawPtr_;
  };
  std::shared_ptr<C_MemoryData> cMemoryData;
  explicit C_Memory(pybind11::int_ &bufferSize,
                    pybind11::str &device,
                    const pybind11::int_ &parallelismSizeThreshold);
  explicit C_Memory();
  ~C_Memory();

  void insert(torch::Tensor &stateCurrent,
              torch::Tensor &stateNext,
              torch::Tensor &reward,
              torch::Tensor &action,
              torch::Tensor &done,
              torch::Tensor &priority,
              torch::Tensor &probability,
              torch::Tensor &weight);
  std::map<std::string, torch::Tensor> get_item(int64_t index);
  void delete_item(int64_t index);
  void set_item(int64_t index,
                torch::Tensor &stateCurrent,
                torch::Tensor &stateNext,
                torch::Tensor &reward,
                torch::Tensor &action,
                torch::Tensor &done,
                torch::Tensor &priority,
                torch::Tensor &probability,
                torch::Tensor &weight);
  std::map<std::string, torch::Tensor> sample(int32_t batchSize,
                                              float_t forceTerminalStateProbability,
                                              bool prioritized);
  void update_transition_priorities(std::vector<int64_t> &indices,
                                    std::vector<torch::Tensor> &newPriorities,
                                    float_t alpha,
                                    float_t beta);
  void update_transition_priorities(torch::Tensor &indices,
                                    torch::Tensor &newPriorities,
                                    float_t alpha,
                                    float_t beta);
  [[nodiscard]] C_MemoryData view() const;
  void initialize(C_MemoryData &viewC_Memory);
  void clear();
  size_t size();
  int64_t numOfTerminalStates();

 private:
  struct TransitionInformation_ {
    torch::Tensor stateCurrent;
    torch::Tensor stateNext;
    torch::Tensor reward;
    torch::Tensor action;
    torch::Tensor done;

    torch::Tensor priority;
    torch::Tensor probability;
    torch::Tensor weight;

    TransitionInformation_();
    ~TransitionInformation_();
  };

  std::deque<std::shared_ptr<TransitionInformation_>> transitionInformationBuffer_;
  std::deque<int64_t> terminalStateIndices_;
  std::map<int64_t, std::shared_ptr<TransitionInformation_>> terminalStateToTransitionBufferMap_;
  std::map<std::shared_ptr<TransitionInformation_>, int64_t> transitionBufferToTerminalStateMap_;
  std::vector<int64_t> loadedIndices_;
  std::deque<float_t> priorities_;
  int64_t parallelismSizeThreshold_ = 8092;
  torch::Device device_ = torch::kCPU;
  int64_t bufferSize_ = 32768;
  int64_t stepCounter_ = 0;
  torch::Tensor maxWeight_ = torch::zeros({}, torch::kFloat32);
  std::map<std::string, torch::DeviceType> deviceMap{
      {"cpu", torch::kCPU},
      {"cuda", torch::kCUDA},
      {"mps", torch::kMPS}
  };
  std::vector<int64_t> get_loaded_indices(int32_t batchSize, float_t forceTerminalStateProbability, bool prioritized);
  std::map<std::string, torch::Tensor> sample_(std::vector<int64_t> &loadedIndicesSlice, int32_t batchSize);
  template<class Container>
  static Container shuffle_indices(int64_t parallelismSizeThreshold, Container &indices);
  void update_transition_priorities_(std::vector<int64_t> &indices,
                                     torch::Tensor &newPriorities,
                                     float_t alpha,
                                     float_t beta);
  static torch::Tensor adjust_dimensions(torch::Tensor &tensor, c10::IntArrayRef &targetDimensions);
  static torch::Tensor compute_probabilities(torch::Tensor &priorities, float_t alpha);
  static torch::Tensor compute_important_sampling_weights(int64_t currentSize,
                                                          torch::Tensor &probability,
                                                          torch::Tensor &maxWeight,
                                                          float_t beta);
};

#endif //RLPACK_C_MEMORY_H
