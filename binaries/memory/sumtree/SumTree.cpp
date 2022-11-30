//
// Created by Kartik Rajeshwaran on 2022-11-01.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

#include "SumTree.h"

SumTree::SumTree(int32_t bufferSize) {
    bufferSize_ = bufferSize;
    leaves_.reserve(bufferSize_);
    sumTree_.reserve(2 * bufferSize_ - 1);
}

SumTree::~SumTree() = default;

SumTree::SumTree() = default;

void SumTree::create_tree(std::deque<float_t> &priorities,
                          std::optional<std::vector<SumTreeNode *>> &children) {
    if (children.has_value()) {
        assert(priorities.size() == children.value().size());
        treeHeight_++;
    }
    std::deque<float_t> prioritiesForTree(priorities.begin(), priorities.end()), prioritiesSum;
    std::vector<SumTreeNode *> childrenForRecursion;
    childrenForRecursion.reserve((prioritiesForTree.size() / 2) + 1);
    if (prioritiesForTree.size() % 2 != 0 && prioritiesForTree.size() != 1) {
        prioritiesForTree.push_back(0);
        if (children.has_value()) {
            auto child = *children;
            child.push_back(nullptr);
        }
    }
    for (int64_t index = 0; index < prioritiesForTree.size(); index = index + 2) {
        auto leftPriority = prioritiesForTree[index];
        auto rightPriority = prioritiesForTree[index + 1];
        auto sum = leftPriority + rightPriority;
        prioritiesSum.push_back(sum);
        if (!children.has_value()) {
            auto parent = new SumTreeNode(nullptr, sum, (int64_t) sumTree_.size() + 2);
            auto leftNode = new SumTreeNode(parent, leftPriority, (int64_t) sumTree_.size(), index);
            auto rightNode = new SumTreeNode(parent, rightPriority,
                                             (int64_t) sumTree_.size() + 1, index + 1);
            parent->set_left_node(leftNode);
            parent->set_right_node(rightNode);
            sumTree_.push_back(leftNode);
            sumTree_.push_back(rightNode);
            sumTree_.push_back(parent);
            leaves_.push_back(leftNode);
            leaves_.push_back(rightNode);
            childrenForRecursion.push_back(parent);
        } else {
            auto child = *children;
            auto leftChild = child[index];
            auto rightChild = child[index + 1];
            auto parent = new SumTreeNode(nullptr,
                                          sum,
                                          (int64_t) sumTree_.size(),
                                          -1,
                                          treeHeight_,
                                          leftChild,
                                          rightChild);
            parent->set_leaf_status(false);
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

void SumTree::reset(int64_t parallelismSizeThreshold) {
    {
        auto enableParallelism = (int64_t) sumTree_.size() > parallelismSizeThreshold;
#pragma omp parallel for if (enableParallelism) default(none) shared(sumTree_) schedule(static)
        for (auto &node: sumTree_) {
            delete node;
        }
    }
    sumTree_.clear();
    leaves_.clear();
    treeHeight_ = 0;
}

int64_t SumTree::sample(float_t seedValue, int64_t currentSize) {
    auto parent = sumTree_.back();
    auto node = traverse(parent, seedValue);
    auto index = node->get_index();
    if (index > currentSize) {
        std::cerr << "WARNING: Larger index than current size was generated " << index << std::endl;
        index = index % currentSize;
    }
    return index;
}

void SumTree::update(int64_t index, float_t value) {
    auto leaf = leaves_[index];
    auto change = value - leaf->get_value();
    auto immediateParent = leaf->get_parent();
    leaf->set_value(value);
    propagate_changes_upwards(immediateParent, change);
}

float_t SumTree::get_cumulative_sum() {
    auto parentNode = sumTree_.back();
    return parentNode->get_value();
}

int64_t SumTree::get_tree_height() {
    auto parent = sumTree_.back();
    return parent->get_tree_level();
}

void SumTree::propagate_changes_upwards(SumTreeNode *node,
                                        float_t change) {
    auto newValue = node->get_value() + change;
    node->set_value(newValue);
    sumTree_[node->get_tree_index()]->set_value(newValue);
    if (!node->is_head()) {
        node = node->get_parent();
        propagate_changes_upwards(node, change);
    }
}

SumTreeNode *SumTree::traverse(SumTreeNode *node, float_t value) {
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