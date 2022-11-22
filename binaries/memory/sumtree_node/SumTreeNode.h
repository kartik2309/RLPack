//
// Created by Kartik Rajeshwaran on 2022-11-01.
//

#ifndef RLPACK_BINARIES_MEMORY_SUMTREE_NODE_SUMTREENODE_H_
#define RLPACK_BINARIES_MEMORY_SUMTREE_NODE_SUMTREENODE_H_

#include <cmath>
#include <stdexcept>

class SumTreeNode {
    /*
 * The class SumTreeNode is a private class which represents a node in Sum-Tree. This is only used
 * when we use proportional prioritization.
 */
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
    SumTreeNode *parent_ = nullptr;
    SumTreeNode *leftNode_ = nullptr;
    SumTreeNode *rightNode_ = nullptr;
    int64_t treeIndex_ = -1;
    int64_t treeLevel_ = 0;
    int64_t index_ = -1;
    float_t value_ = 0;
    bool isLeaf_ = true;
};

#endif//RLPACK_BINARIES_MEMORY_SUMTREE_NODE_SUMTREENODE_H_
