//
// Created by Kartik Rajeshwaran on 2022-11-01.
//

#include "SumTreeNode.h"


SumTreeNode::SumTreeNode(SumTreeNode *parent,
                         float_t value,
                         int64_t treeIndex,
                         int64_t index,
                         int64_t treeLevel,
                         SumTreeNode *leftNode,
                         SumTreeNode *rightNode) {
    parent_ = parent;
    leftNode_ = leftNode;
    rightNode_ = rightNode;
    treeIndex_ = treeIndex;
    index_ = index;
    treeLevel_ = treeLevel;
    value_ = value;

    if (leftNode_ != nullptr || rightNode_ != nullptr) {
        isLeaf_ = false;
    }
}

SumTreeNode::~SumTreeNode() = default;

void SumTreeNode::set_value(float_t newValue) {
    value_ = newValue;
}

[[maybe_unused]] void SumTreeNode::remove_left_node() {
    leftNode_ = nullptr;
}

[[maybe_unused]] void SumTreeNode::remove_right_node() {
    rightNode_ = nullptr;
}

void SumTreeNode::set_left_node(SumTreeNode *node) {
    leftNode_ = node;
    if (leftNode_ != nullptr || rightNode_ != nullptr) {
        isLeaf_ = false;
    }
}

void SumTreeNode::set_right_node(SumTreeNode *node) {
    if (leftNode_ == nullptr) {
        throw std::runtime_error("Tried to add right node before setting left node!");
    }
    rightNode_ = node;
    if (leftNode_ != nullptr || rightNode_ != nullptr) {
        isLeaf_ = false;
    }
}

float_t SumTreeNode::get_value() const {
    return value_;
}

int64_t SumTreeNode::get_tree_index() const {
    return treeIndex_;
}

int64_t SumTreeNode::get_index() const {
    return index_;
}

int64_t SumTreeNode::get_tree_level() const {
    return treeLevel_;
}

SumTreeNode *SumTreeNode::get_parent() {
    return parent_;
}

SumTreeNode *SumTreeNode::get_left_node() {
    return leftNode_;
}

SumTreeNode *SumTreeNode::get_right_node() {
    return rightNode_;
}

bool SumTreeNode::is_leaf() const {
    return isLeaf_;
}

bool SumTreeNode::is_head() {
    if (parent_ != nullptr) {
        return false;
    }
    return true;
}

void SumTreeNode::set_parent_node(SumTreeNode *parent) {
    parent_ = parent;
}

void SumTreeNode::set_leaf_status(bool isLeaf) {
    isLeaf_ = isLeaf;
}
