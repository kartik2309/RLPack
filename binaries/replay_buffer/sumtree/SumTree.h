
#ifndef RLPACK_BINARIES_REPLAY_BUFFER_SUMTREE_SUMTREE_H_
#define RLPACK_BINARIES_REPLAY_BUFFER_SUMTREE_SUMTREE_H_

#include <cassert>
#include <deque>
#include <iostream>
#include <optional>
#include <vector>

#include "../sumtree_node/SumTreeNode.h"

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup replay_buffer_group replay_buffer
 * @brief Memory module is the C++ backend for rlpack._C.replay_buffer.ReplayBuffer class. Heavier workloads have been optimized
 * with multithreading with OpenMP and CUDA (if CUDA compatible device is found).
 * @{
 */
 /*!
  * @brief The class SumTree is a class which represents the Sum-Tree which is used in proportional
  * prioritization. It implements all the methods necessary to create the Sum-Tree and sample from it.
  */
class SumTree {
public:
    explicit SumTree(int32_t bufferSize);
    SumTree();
    ~SumTree();
    void create_tree(std::deque<float_t> &priorities,
                     std::optional<std::vector<SumTreeNode *>> &children);
    void reset(int64_t parallelismSizeThreshold = 4096);
    int64_t sample(float_t seedValue, int64_t currentSize);
    void update(int64_t index, float_t value);
    [[maybe_unused]] float_t get_cumulative_sum();
    int64_t get_tree_height();

private:
    //! A vector to store the pointers to dynamically allocated SumTreeNode nodes.
    std::vector<SumTreeNode *> sumTree_;
    //! A vector to store the pointers to dynamically allocated SumTreeNode leaves.
    std::vector<SumTreeNode *> leaves_;
    //! Attribute to store the buffer size. If not initialised, when using SumTree::SumTree(), will be set to 32768.
    int64_t bufferSize_ = 32768;
    //! Attribute to store the tree height.
    int64_t treeHeight_ = 0;

    void propagate_changes_upwards(SumTreeNode *node, float_t change);
    SumTreeNode *traverse(SumTreeNode *node, float_t value);
};
/*!
 * @} @I{ // End group memory_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_REPLAY_BUFFER_SUMTREE_SUMTREE_H_
