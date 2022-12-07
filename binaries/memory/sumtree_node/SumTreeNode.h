
#ifndef RLPACK_BINARIES_MEMORY_SUMTREE_NODE_SUMTREENODE_H_
#define RLPACK_BINARIES_MEMORY_SUMTREE_NODE_SUMTREENODE_H_

#include <cmath>
#include <stdexcept>

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup memory_group memory
 * @brief Memory module is the C++ backend for rlpack._C.memory.Memory class. Heavier workloads have been optimized
 * with multithreading with OpenMP and CUDA (if CUDA compatible device is found).
 * @{
 */
/*!
  * @brief The class SumTreeNode is a private class which represents a node in Sum-Tree. This is only used
  * when we use proportional prioritization.
  */
class SumTreeNode {
public:
    SumTreeNode(SumTreeNode *parent,
                float_t value,
                int64_t treeIndex = -1,
                int64_t index = -1,
                int64_t treeLevel = 0,
                SumTreeNode *leftNode = nullptr,
                SumTreeNode *rightNode = nullptr);
    ~SumTreeNode();

public:
    void set_value(float_t newValue);
    [[maybe_unused]] void remove_left_node();
    [[maybe_unused]] void remove_right_node();
    void set_left_node(SumTreeNode *node);
    void set_right_node(SumTreeNode *node);
    void set_parent_node(SumTreeNode *parent);
    void set_leaf_status(bool isLeaf);
    [[nodiscard]] float_t get_value() const;
    [[nodiscard]] int64_t get_tree_index() const;
    [[nodiscard]] int64_t get_index() const;
    [[nodiscard]] int64_t get_tree_level() const;
    SumTreeNode *get_parent();
    SumTreeNode *get_left_node();
    SumTreeNode *get_right_node();
    [[nodiscard]] bool is_leaf() const;
    bool is_head();

private:
    //! The pointer to parent node.
    SumTreeNode *parent_ = nullptr;
    //! The pointer to left child node.
    SumTreeNode *leftNode_ = nullptr;
    //! The pointer to right child node.
    SumTreeNode *rightNode_ = nullptr;
    //! The tree index of current node. This represents the index in of the node in container buffer.
    int64_t treeIndex_ = -1;
    //! The tree level of current node. This represents the height of the node in the current tree.
    int64_t treeLevel_ = 0;
    //! The index value of current node. This is only valid for leaf node and represents the index of the node from left.
    int64_t index_ = -1;
    //! The value of current node. Value of each node is the sum of its children's value.
    float_t value_ = 0;
    //! The boolean flag indicating if the current node is a leaf node or not.
    bool isLeaf_ = true;
};
/*!
 * @} @I{ // End group memory_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_MEMORY_SUMTREE_NODE_SUMTREENODE_H_
