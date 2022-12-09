
#include "SumTreeNode.h"


SumTreeNode::SumTreeNode(SumTreeNode *parent,
                         float_t value,
                         int64_t treeIndex,
                         int64_t index,
                         int64_t treeLevel,
                         SumTreeNode *leftNode,
                         SumTreeNode *rightNode) {
    /*!
     * Class constructor for SumTreeNode. Using this constructor we can set the pointers correctly to link the
     * node with its right children and parents (if any).
     *
     * @param *parent : The pointer to the parent node. Can be set to nullptr if parent hasn't been allocated yet.
     * @param value : The value of the node. Value of each node is the sum of its children's value.
     * @param treeIndex : The tree index of a node representing the index in of the node in container buffer.
     * @param index : The index of the leaf node. It should be set to -1 if the node is not a leaf node.
     * @param treeLevel : The tree level of current node representing the height of the node in the current tree.
     * @param *leftNode : The pointer to left child node.
     * @param *rightNode : The pointer to right child node.
     */
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

/*!
 * Sum-Tree Node default destructor.
 */
SumTreeNode::~SumTreeNode() = default;

void SumTreeNode::set_value(float_t newValue) {
    /*!
     * Sets the new value in the node.
     *
     * @param newValue : The new value to be set in the node.
     */
    value_ = newValue;
}

[[maybe_unused]] void SumTreeNode::remove_left_node() {
    /*!
     * Removes the left node. This will only de-link the nodes and will not de-allocate/free any memory.
     */
    leftNode_ = nullptr;
}

[[maybe_unused]] void SumTreeNode::remove_right_node() {
    /*!
     * Removes the right node. This will only de-link the nodes and will not de-allocate/free any memory.
     */
    rightNode_ = nullptr;
}

void SumTreeNode::set_left_node(SumTreeNode *node) {
    /*!
     * Sets the left child node for a given node. This must be set first before setting the right child node.
     *
     * @param *node : The pointer to the left node to be linked.
     */
    leftNode_ = node;
    if (leftNode_ != nullptr || rightNode_ != nullptr) {
        isLeaf_ = false;
    }
}

void SumTreeNode::set_right_node(SumTreeNode *node) {
    /*!
     * Sets the right child node for a given node. This will throw `std::runtime_error` if left node has not been set.
     *
     * @param *node : The pointer to the right node to be linked.
     *
     */
    if (leftNode_ == nullptr) {
        throw std::runtime_error("Tried to add right node before setting left node!");
    }
    rightNode_ = node;
    if (rightNode_ != nullptr) {
        isLeaf_ = false;
    }
}

void SumTreeNode::set_parent_node(SumTreeNode *parent) {
    /*!
     * Sets the parent of the given node.
     *
     * @param *parent : The pointer to the parent node to be linked.
     */
    parent_ = parent;
}

void SumTreeNode::set_leaf_status(bool isLeaf) {
    /*!
     * Sets the leaf status of the given node.
     *
     * @param *isLeaf : Boolean flag to be used to update.
     */
    isLeaf_ = isLeaf;
}

float_t SumTreeNode::get_value() const {
    /*!
     * Get the float value of the current node; from SumTreeNode::value_.
     *
     * @return : The float value of the node.
     */
    return value_;
}

int64_t SumTreeNode::get_tree_index() const {
    /*!
     * Get the tree index value of the current node; from SumTreeNode::treeIndex_.
     *
     * @return : The tree index value.
     */
    return treeIndex_;
}

int64_t SumTreeNode::get_index() const {
    /*!
     * Get the index value of the current node; from SumTreeNode::index_.
     *
     * @return : The index value of the of leaf nodes, -1 for other nodes.
     */
    return index_;
}

int64_t SumTreeNode::get_tree_level() const {
    /*!
     * Get the tree level value of the current node, i.e. the node's height; from SumTreeNode::treeLevel_.
     *
     * @return : The tree level value, 1 indicating leaf level.
     */
    return treeLevel_;
}

SumTreeNode *SumTreeNode::get_parent() {
    /*!
     * Get the pointer to parent node.
     *
     * @return : The pointer to the parent node.
     */
    return parent_;
}

SumTreeNode *SumTreeNode::get_left_node() {
    /*!
     * Get the pointer to left child node.
     *
     * @return : The pointer to the left child node.
     */
    return leftNode_;
}

SumTreeNode *SumTreeNode::get_right_node() {
    /*!
     * Get the pointer to right child node.
     *
     * @return : The pointer to the right child node.
     */
    return rightNode_;
}

bool SumTreeNode::is_leaf() const {
    /*!
     * Get the status of the current node to check if the node is a leaf or not. It returns true for leaf nodes.
     *
     * @return : Flag indicating if the node is a leaf or not.
     */
    return isLeaf_;
}

bool SumTreeNode::is_head() {
    /*!
     * Get the status of the current node to check if the node is the root node.
     *
     * @return : Flag indicating if the node is the head/root or not.
     */
    if (parent_ != nullptr) {
        return false;
    }
    return true;
}
