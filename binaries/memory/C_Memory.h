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
    void set_transition_information_reference(std::shared_ptr<torch::Tensor> &stateCurrent,
                                              std::shared_ptr<torch::Tensor> &stateNext,
                                              std::shared_ptr<torch::Tensor> &reward,
                                              std::shared_ptr<torch::Tensor> &action,
                                              std::shared_ptr<torch::Tensor> &done);
    void set_transition_information_reference(int64_t index,
                                              std::shared_ptr<torch::Tensor> &stateCurrent,
                                              std::shared_ptr<torch::Tensor> &stateNext,
                                              std::shared_ptr<torch::Tensor> &reward,
                                              std::shared_ptr<torch::Tensor> &action,
                                              std::shared_ptr<torch::Tensor> &done);
    void set_terminal_state_index_reference(const std::shared_ptr<int64_t> &terminalStateIndex);
    void set_terminal_state_index_reference(int64_t index,
                                            const std::shared_ptr<int64_t> &terminalStateIndex);
    void delete_transition_information_reference(int64_t index);
    void delete_terminal_state_index_reference(int64_t terminalStateIndex);
    void initialize_data(std::map<std::string, std::vector<torch::Tensor>> &transitionInformationMap,
                         std::map<std::string, std::vector<int64_t>> &terminalStateIndicesMap);
    size_t _size();
    int64_t _num_of_terminal_states();
    [[nodiscard]] int64_t get_buffer_size() const;
    [[nodiscard]] int64_t get_parallelism_size_threshold() const;
   private:
    std::map<std::string, std::deque<std::shared_ptr<torch::Tensor>>> transitionInformationMap_;
    std::deque<std::shared_ptr<int64_t>> terminalStateIndexReferences_;
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
              bool isTerminalState);
  std::map<std::string, torch::Tensor> get_item(int64_t index);
  void delete_item(int64_t index);
  void set_item(int64_t index,
                torch::Tensor &stateCurrent,
                torch::Tensor &stateNext,
                torch::Tensor &reward,
                torch::Tensor &action,
                torch::Tensor &done,
                bool isTerminalState);
  std::map<std::string, torch::Tensor> sample(int32_t batchSize, float_t forceTerminalStateProbability);
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

    TransitionInformation_();
    ~TransitionInformation_();
  };

  std::deque<std::shared_ptr<TransitionInformation_>> transitionInformationBuffer_;
  std::deque<int64_t> terminalStateIndices_;
  std::map<int64_t, std::shared_ptr<TransitionInformation_>> terminalStateToTransitionBufferMap_;
  std::map<std::shared_ptr<TransitionInformation_>, int64_t> transitionBufferToTerminalStateMap_;
  std::vector<int64_t> loadedIndices_;
  int64_t parallelismSizeThreshold_ = 8092;
  torch::Device device_ = torch::kCPU;
  int64_t bufferSize_ = 32768;
  int64_t stepCounter_ = 0;
  std::map<std::string, torch::DeviceType> deviceMap{
      {"cpu", torch::kCPU},
      {"cuda", torch::kCUDA},
      {"mps", torch::kMPS}
  };

  std::vector<int64_t> shuffle_loaded_indices(int64_t parallelismSizeThreshold);
  static torch::Tensor adjust_dimensions(torch::Tensor &tensor, c10::IntArrayRef &targetDimensions);
};

#endif //RLPACK_C_MEMORY_H
