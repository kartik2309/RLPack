//
// Created by Kartik Rajeshwaran on 2022-11-01.
//

#ifndef RLPACK_BINARIES_MEMORY_SUMTREE_SUMTREE_H_
#define RLPACK_BINARIES_MEMORY_SUMTREE_SUMTREE_H_

#include <cassert>
#include <deque>
#include <iostream>
#include <optional>
#include <vector>

#include "../sumtree_node/SumTreeNode.h"

class SumTree {
    /*
   * The class SumTree is a private class which represents the Sum-Tree which is used in proportional
   * prioritization. It implements all the methods necessary to create the Sum-Tree and sample from it.
   */
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
    std::vector<SumTreeNode *> sumTree_;
    std::vector<SumTreeNode *> leaves_;
    int64_t bufferSize_ = 32768;
    int64_t treeHeight_ = 0;

    void propagate_changes_upwards(SumTreeNode *node, float_t change);
    SumTreeNode *traverse(SumTreeNode *node, float_t value);
};

#endif//RLPACK_BINARIES_MEMORY_SUMTREE_SUMTREE_H_
