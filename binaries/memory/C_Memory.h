//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#ifndef RLPACK_C_MEMORY_H
#define RLPACK_C_MEMORY_H

#include <pybind11/pybind11.h>
#include <random>
#include <omp.h>
#include <torch/extension.h>
#include <vector>
#include <deque>
#include <algorithm>

class C_Memory {
  /*
   * The class C_Memory is the C++ backend for memory-buffer used in algorithms that stores transitions in a buffer.
   * This class contains optimized routines to support Python front-end of Memory class.
   */
 public:

  struct C_MemoryData {
    /*
     * The class C_MemoryData keeps the references to data that is associated with C_Memory. This class
     * implements the functions necessary to retrieve the data by de-referencing the data associated with C_Memory.
     */

    C_MemoryData();
    ~C_MemoryData();

    std::map<std::string, std::deque<torch::Tensor>> dereference_transition_information();
    [[nodiscard]] std::map<std::string, std::deque<int64_t>> dereference_terminal_state_indices() const;
    [[nodiscard]] std::map<std::string, std::deque<float_t>> dereference_priorities() const;
    void set_transition_information_references(std::deque<torch::Tensor> *&statesCurrent,
                                               std::deque<torch::Tensor> *&statesNext,
                                               std::deque<torch::Tensor> *&rewards,
                                               std::deque<torch::Tensor> *&actions,
                                               std::deque<torch::Tensor> *&dones,
                                               std::deque<torch::Tensor> *&priorities,
                                               std::deque<torch::Tensor> *&probabilities,
                                               std::deque<torch::Tensor> *&weights);
    void set_transition_information_references(std::string &key,
                                               std::deque<torch::Tensor> *&reference);
    void set_terminal_state_indices_reference(std::deque<int64_t> *&terminalStateIndicesReference);
    void set_priorities_reference(std::deque<float_t> *&prioritiesFloatReference);

   private:
    std::map<std::string, std::deque<torch::Tensor> *> transitionInformationReference_;
    std::deque<int64_t> *terminalIndicesReference_ = nullptr;
    std::deque<float_t> *prioritiesFloatReference_ = nullptr;
  };
  std::shared_ptr<C_MemoryData> cMemoryData;

  explicit C_Memory(const pybind11::int_ &bufferSize,
                    const pybind11::str &device,
                    const pybind11::int_ &prioritizationStrategyCode);
  explicit C_Memory();
  ~C_Memory();

  void insert(torch::Tensor &stateCurrent,
              torch::Tensor &stateNext,
              torch::Tensor &reward,
              torch::Tensor &action,
              torch::Tensor &done,
              torch::Tensor &priority,
              torch::Tensor &probability,
              torch::Tensor &weight,
              bool isTerminalState);
  std::map<std::string, torch::Tensor> get_item(int64_t index);
  void set_item(int64_t index,
                torch::Tensor &stateCurrent,
                torch::Tensor &stateNext,
                torch::Tensor &reward,
                torch::Tensor &action,
                torch::Tensor &done,
                torch::Tensor &priority,
                torch::Tensor &probability,
                torch::Tensor &weight,
                bool isTerminalState);
  void delete_item(int64_t index);
  std::map<std::string, torch::Tensor> sample(int32_t batchSize,
                                              float_t forceTerminalStateProbability,
                                              int64_t parallelismSizeThreshold,
                                              float_t alpha = 0.0,
                                              float_t beta = 0.0,
                                              int64_t numSegments = 0);
  void update_priorities(torch::Tensor &randomIndices,
                         torch::Tensor &newPriorities,
                         torch::Tensor &newProbabilities,
                         torch::Tensor &newWeights);
  [[nodiscard]] C_MemoryData view() const;
  void initialize(C_MemoryData &viewC_Memory);
  void clear();
  size_t size();
  int64_t num_terminal_states();
  int64_t tree_height();

 private:
  struct SumTreeNode_ {
    /*
     * The class SumTreeNode_ is a private class which represents a node in Sum-Tree. This is only used
     * when we use proportional prioritization.
     */
    SumTreeNode_(SumTreeNode_ *parent,
                 float_t value,
                 int64_t treeIndex = -1,
                 int64_t index = -1,
                 int64_t treeLevel = 0,
                 SumTreeNode_ *leftNode = nullptr,
                 SumTreeNode_ *rightNode = nullptr);
    ~SumTreeNode_();

    void set_value(float_t newValue);
    [[maybe_unused]] void remove_left_node();
    [[maybe_unused]] void remove_right_node();
    void set_left_node(SumTreeNode_ *node);
    void set_right_node(SumTreeNode_ *node);
    void set_parent_node(SumTreeNode_ *parent);
    void set_leaf_status(bool isLeaf);
    [[nodiscard]] float_t get_value() const;
    [[nodiscard]] int64_t get_tree_index() const;
    [[nodiscard]] int64_t get_index() const;
    [[nodiscard]] int64_t get_tree_level() const;
    SumTreeNode_ *get_parent();
    SumTreeNode_ *get_left_node();
    SumTreeNode_ *get_right_node();
    [[nodiscard]] bool is_leaf() const;
    bool is_head();

   private:
    SumTreeNode_ *parent_ = nullptr;
    SumTreeNode_ *leftNode_ = nullptr;
    SumTreeNode_ *rightNode_ = nullptr;
    int64_t treeIndex_ = -1;
    int64_t treeLevel_ = 0;
    int64_t index_ = -1;
    float_t value_ = 0;
    bool isLeaf_ = true;

  };

  struct SumTree_ {
    /*
     * The class SumTree_ is a private class which represents the Sum-Tree which is used in proportional
     * prioritization. It implements all the methods necessary to create the Sum-Tree and sample from it.
     */
    explicit SumTree_(int32_t bufferSize);
    SumTree_();
    ~SumTree_();
    void create_tree(std::deque<float_t> &priorities,
                     std::optional<std::vector<SumTreeNode_ *>> &children);
    void reset(int64_t parallelismSizeThreshold = 4096);
    int64_t sample(float_t seedValue, int64_t currentSize);
    void update(int64_t index, float_t value);
    [[maybe_unused]] float_t get_cumulative_sum();
    int64_t get_tree_height();
   private:
    std::vector<SumTreeNode_ *> sumTree_;
    std::vector<SumTreeNode_ *> leaves_;
    int64_t bufferSize_ = 32768;
    int64_t treeHeight_ = 0;

    void propagate_changes_upwards(SumTreeNode_ *node, float_t change);
    C_Memory::SumTreeNode_ *traverse(C_Memory::SumTreeNode_ *node, float_t value);
  };

  std::deque<torch::Tensor> statesCurrent_;
  std::deque<torch::Tensor> statesNext_;
  std::deque<torch::Tensor> rewards_;
  std::deque<torch::Tensor> actions_;
  std::deque<torch::Tensor> dones_;
  std::deque<torch::Tensor> priorities_;
  std::deque<torch::Tensor> probabilities_;
  std::deque<torch::Tensor> weights_;
  std::deque<int64_t> terminalStateIndices_;
  std::deque<float_t> prioritiesFloat_;
  std::vector<int64_t> loadedIndices_;
  std::shared_ptr<SumTree_> sumTreeSharedPtr_;
  torch::Device device_ = torch::kCPU;
  int64_t bufferSize_ = 32768;
  int64_t stepCounter_ = 0;
  int32_t prioritizationStrategyCode_ = 0;
  std::map<std::string, torch::DeviceType> deviceMap_{
      {"cpu", torch::kCPU},
      {"cuda", torch::kCUDA},
      {"mps", torch::kMPS}
  };
  static float_t get_cumulative_sum_of_deque(const std::deque<float_t> &prioritiesFloat,
                                             int64_t parallelismSizeThreshold);
  static std::vector<int64_t> get_shuffled_vector(std::vector<int64_t> &loadedIndices,
                                                  int64_t parallelismSizeThreshold);
  static std::vector<float_t> get_priority_seeds(float_t cumulativeSum, int64_t parallelismSizeThreshold);
  static torch::Tensor compute_probabilities(torch::Tensor &priorities, float_t alpha);
  static torch::Tensor compute_important_sampling_weights(torch::Tensor &probabilities,
                                                          int64_t currentSize,
                                                          float_t beta);
  static std::vector<int64_t> compute_quantile_segment_indices(int64_t numSegments,
                                                               const std::deque<float_t> &prioritiesFloat,
                                                               const std::function<float_t(const std::deque<
                                                                   float_t> &,
                                                                                           int64_t)> &cumulativeSumFunction,
                                                               int64_t parallelismSizeThreshold);
  static void arg_mergesort_for_pair_vector(std::vector<std::pair<float_t, int64_t>> &priorityFloatIndicesPair,
                                            int64_t begin,
                                            int64_t end,
                                            bool enableParallelism);
  static void arg_merge_for_pair_vector(std::vector<std::pair<float_t, int64_t>> &priorityFloatIndicesPair,
                                        int64_t left,
                                        int64_t mid,
                                        int64_t right,
                                        bool enableParallelism);
};

#endif //RLPACK_C_MEMORY_H