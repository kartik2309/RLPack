//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_DCQN1DOPTIONS_H
#define RLPACK_DCQN1DOPTIONS_H

#include <torch/nn.h>
#include "../../../utils/Base/ModelBase/ModelBase.h"
#include "../../../utils/Base/Options/ModelOptions/ModelOptionsBase.h"

namespace model::dqn {
    class Dcqn1dOptions : public ModelOptionsBase {
        int32_t sequenceLength_{};
        std::vector<int32_t> channels_;
        std::vector<int32_t> kernelSizesConv_;
        std::vector<int32_t> strideSizesConv_;
        std::vector<int32_t> dilationSizesConv_;
        std::optional<std::vector<int32_t>> kernelSizesPool_ = std::optional<std::vector<int32_t>>();
        std::optional<std::vector<int32_t>> strideSizesPool_ = std::optional<std::vector<int32_t>>();
        std::optional<std::vector<int32_t>> dilationSizesPool_ = std::optional<std::vector<int32_t>>();
        float_t dropout_ = 0.5;
        bool usePadding_ = true;
    public:
        Dcqn1dOptions();

        Dcqn1dOptions(
                int32_t sequenceLength, std::vector<int32_t> &channels, std::vector<int32_t> &kernelSizesConv,
                std::vector<int32_t> &strideSizesConv, std::vector<int32_t> &dilationSizesConv, int32_t numActions,
                const std::optional<std::vector<int32_t>> &kernelSizesPool = std::optional<std::vector<int32_t>>(),
                const std::optional<std::vector<int32_t>> &strideSizesPool = std::optional<std::vector<int32_t>>(),
                const std::optional<std::vector<int32_t>> &dilationSizesPool = std::optional<std::vector<int32_t>>(),
                std::optional<std::shared_ptr<ActivationBase>> activation = std::optional<std::shared_ptr<ActivationBase>>(),
                float_t dropout = 0.5, bool usePadding = true
        );

        ~Dcqn1dOptions();

        void sequence_length(int32_t sequenceLength);

        void channels(std::vector<int32_t> &channels);

        void conv_kernel_sizes(std::vector<int32_t> &kernelSizesConv);

        void conv_stride_sizes(std::vector<int32_t> &strideSizesConv);

        void conv_dilation_sizes(std::vector<int32_t> &dilationSizesConv);

        void pool_kernel_sizes(std::optional<std::vector<int32_t>> &kernelSizesPool);

        void pool_stride_sizes(std::optional<std::vector<int32_t>> &strideSizesPool);

        void pool_dilation_sizes(std::optional<std::vector<int32_t>> &dilationSizesPool);

        void dropout(float_t dropout);

        void use_padding(bool usePadding);

        [[nodiscard]] int32_t get_sequence_length() const;

        std::vector<int32_t> get_channels();

        std::vector<int32_t> get_conv_kernel_sizes();

        std::vector<int32_t> get_conv_strides_sizes();

        std::vector<int32_t> get_conv_dilation_sizes();

        std::optional<std::vector<int32_t>> get_pool_kernel_sizes();

        std::optional<std::vector<int32_t>> get_pool_strides_sizes();

        std::optional<std::vector<int32_t>> get_pool_dilation_sizes();

        [[nodiscard]] float_t get_dropout() const;

        [[nodiscard]] bool get_use_padding() const;

    };


}

#endif //RLPACK_DCQN1DOPTIONS_H
