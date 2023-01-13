
#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

#include "SumTree.h"

SumTree::SumTree(int32_t bufferSize) {
    /*!
     * The class constructor will bufferSize. This constructor will reserve memory according the buffer size passed.
     * @param bufferSize : Buffer Size to be elements to be used to create the sum tree.
     */
    bufferSize_ = bufferSize;
    leaves_.reserve(bufferSize_);
    sumTree_.reserve(2 * bufferSize_ - 1);
}

/*!
 * Sum Tree default destructor.
 */
SumTree::~SumTree() = default;

/*!
 * Sum Tree default constructor.
 */
SumTree::SumTree() = default;

void SumTree::create_tree(std::deque<float_t> &priorities,
                          std::optional<std::vector<SumTreeNode *>> &children) {
    /*!
     * Builds the sum-tree data structure on the fly.
     * @param priorities : The priorities which are to be used to create the sum-tree. Priorities are used to create
     *  the leaves for the sum-tree.
     * @param children: The children at a tree level. This is an optional argument and is only used for recursion.
     */
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
            auto childrenValue = *children;
            childrenValue.push_back(nullptr);
            children = childrenValue;
        }
    }
    for (int64_t index = 0; index < prioritiesForTree.size(); index = index + 2) {
        auto leftPriority = prioritiesForTree[index];
        auto rightPriority = prioritiesForTree[index + 1];
        auto sum = leftPriority + rightPriority;
        prioritiesSum.push_back(sum);
        if (not children.has_value()) {
            auto parent = new SumTreeNode(nullptr, sum, (int64_t) sumTree_.size() + 2);
            auto leftNode = new SumTreeNode(parent, leftPriority, (int64_t) sumTree_.size(), index);
            auto rightNode = new SumTreeNode(parent,
                                             rightPriority,
                                             (int64_t) sumTree_.size() + 1,
                                             index + 1);
            parent->set_left_node(leftNode);
            parent->set_right_node(rightNode);
            sumTree_.push_back(leftNode);
            sumTree_.push_back(rightNode);
            sumTree_.push_back(parent);
            leaves_.push_back(leftNode);
            leaves_.push_back(rightNode);
            childrenForRecursion.push_back(parent);
        } else {
            auto childrenValue = *children;
            auto leftChild = childrenValue[index];
            auto rightChild = childrenValue[index + 1];
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
    /*!
     * Resets the sum tree by de-allocating all the SumTreeNode objects and clearing the relevant vectors.
     * @param parallelismSizeThreshold : The length threshold for @ref sumTree_ after which the de-allocation
     *  is done parallely with OpenMP
     */
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
    /*!
     * Performs sampling by traversing the sum tree with seed value. The argument `currentSize` is
     * requested since this helps perform safety check for final sampled index to ensure that it is valid.
     *
     * @param seedValue : The seed value which is used to traverse the sum tree.
     * @param currentSize : The current size of the priorities from which the sum tree was created.
     * @return The sampled Index.
     */
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
    /*!
     * Updates the change sum tree by for a given leaf index by the given value. The changes are propagated
     * upwards till the root.
     * @param index : The leaf index to be updated.
     * @param value : The new value for leaf node at the specified index.
     */
    auto leaf = leaves_[index];
    auto change = value - leaf->get_value();
    auto immediateParent = leaf->get_parent();
    leaf->set_value(value);
    propagate_changes_upwards(immediateParent, change);
}

float_t SumTree::get_cumulative_sum() {
    /*!
     * Returns the cumulative sum of the sum tree. This is the value of the root node.
     *
     * @return The cumulative sum of give priority buffer.
     */
    auto parentNode = sumTree_.back();
    return parentNode->get_value();
}

int64_t SumTree::get_tree_height() {
    /*!
     * Returns the height of the tree that is created. Height calculation includes the leaves level.
     *
     * @return The height of the tree.
     */
    auto parent = sumTree_.back();
    return parent->get_tree_level();
}

void SumTree::propagate_changes_upwards(SumTreeNode *node,
                                        float_t change) {
    /*!
     * Propagates changes upwards by making the change to each node recursively until root node.
     *
     * @param *node : The node pointer on which the change is to be applied.
     * @param change: The change of value which is to be applied on the given node.
     */
    auto newValue = node->get_value() + change;
    node->set_value(newValue);
    sumTree_[node->get_tree_index()]->set_value(newValue);
    if (!node->is_head()) {
        node = node->get_parent();
        propagate_changes_upwards(node, change);
    }
}

SumTreeNode *SumTree::traverse(SumTreeNode *node, float_t value) {
    /*!
     * The traversal of the tree from the root node given a value. The value must be between 0 and root node's value,
     * i.e. cumulative sum. A valid random value will select an index based on weighted uniform distribution.
     *
     * @param *node : The node pointer on which the change is to be applied.
     * @param value : A valid random value between 0 and root node's value.
     */
    if (!node->is_leaf()) {
        auto leftNode = node->get_left_node();
        auto rightNode = node->get_right_node();
        if (leftNode->get_value() >= value or rightNode == nullptr) {
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