//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Dcqn1dOptions.h"

namespace model {

    dqn::Dcqn1dOptions::Dcqn1dOptions() = default;

    dqn::Dcqn1dOptions::Dcqn1dOptions(
            int32_t sequenceLength, std::vector<int32_t> &channels,
            std::vector<int32_t> &kernelSizesConv,
            std::vector<int32_t> &strideSizesConv,
            std::vector<int32_t> &dilationSizesConv, int32_t numActions,
            const std::optional<std::vector<int32_t>> &kernelSizesPool,
            const std::optional<std::vector<int32_t>> &strideSizesPool,
            const std::optional<std::vector<int32_t>> &dilationSizesPool,
            std::optional<std::shared_ptr<ActivationBase>> activation, float_t dropout, bool usePadding
    ) : ModelOptionsBase(numActions, activation) {
        sequenceLength_ = sequenceLength;
        channels_ = channels;
        kernelSizesConv_ = kernelSizesConv;
        strideSizesConv_ = strideSizesConv;
        dilationSizesConv_ = dilationSizesConv;
        numAction_ = numActions;
        kernelSizesPool_ = kernelSizesPool;
        strideSizesPool_ = strideSizesPool;
        dilationSizesPool_ = dilationSizesPool;
        activation_ = activation;
        dropout_ = dropout;
        usePadding_ = usePadding;
    }

    void dqn::Dcqn1dOptions::sequence_length(int32_t sequenceLength) {
        sequenceLength_ = sequenceLength;
    }

    void dqn::Dcqn1dOptions::channels(std::vector<int32_t> &channels) {
        channels_ = channels;
    }

    void dqn::Dcqn1dOptions::conv_kernel_sizes(std::vector<int32_t> &kernelSizesConv) {
        kernelSizesConv_ = kernelSizesConv;
    }

    void dqn::Dcqn1dOptions::conv_stride_sizes(std::vector<int32_t> &strideSizesConv) {
        strideSizesConv_ = strideSizesConv;
    }

    void dqn::Dcqn1dOptions::conv_dilation_sizes(std::vector<int32_t> &dilationSizesConv) {
        dilationSizesConv_ = dilationSizesConv;
    }

    void dqn::Dcqn1dOptions::pool_kernel_sizes(std::optional<std::vector<int32_t>> &kernelSizesPool) {
        kernelSizesPool_ = kernelSizesPool;
    }

    void dqn::Dcqn1dOptions::pool_stride_sizes(std::optional<std::vector<int32_t>> &strideSizePool) {
        strideSizesPool_ = strideSizePool;
    }

    void dqn::Dcqn1dOptions::pool_dilation_sizes(std::optional<std::vector<int32_t>> &dilationSizesPool) {
        dilationSizesPool_ = dilationSizesPool;
    }

    void dqn::Dcqn1dOptions::dropout(float_t dropout) {
        dropout_ = dropout;
    }

    void dqn::Dcqn1dOptions::use_padding(bool usePadding) {
        usePadding_ = usePadding;
    }

    int32_t dqn::Dcqn1dOptions::get_sequence_length() const {
        return sequenceLength_;
    }

    std::vector<int32_t> dqn::Dcqn1dOptions::get_channels() {
        return channels_;
    }

    std::vector<int32_t> dqn::Dcqn1dOptions::get_conv_kernel_sizes() {
        return kernelSizesConv_;
    }

    std::vector<int32_t> dqn::Dcqn1dOptions::get_conv_strides_sizes() {
        return strideSizesConv_;
    }

    std::vector<int32_t> dqn::Dcqn1dOptions::get_conv_dilation_sizes() {
        return dilationSizesConv_;
    }

    std::optional<std::vector<int32_t>> dqn::Dcqn1dOptions::get_pool_kernel_sizes() {
        return kernelSizesPool_;
    }

    std::optional<std::vector<int32_t>> dqn::Dcqn1dOptions::get_pool_strides_sizes() {
        return strideSizesPool_;
    }

    std::optional<std::vector<int32_t>> dqn::Dcqn1dOptions::get_pool_dilation_sizes() {
        return dilationSizesPool_;
    }

    float_t dqn::Dcqn1dOptions::get_dropout() const {
        return dropout_;
    }

    bool dqn::Dcqn1dOptions::get_use_padding() const {
        return usePadding_;
    }

    dqn::Dcqn1dOptions::~Dcqn1dOptions() = default;

}

