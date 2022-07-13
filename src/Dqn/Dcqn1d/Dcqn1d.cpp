//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#include "Dcqn1d.h"

namespace model::dqn {

    Dcqn1d::Dcqn1d(
            int32_t sequenceLength, std::vector<int32_t> &channels, std::vector<int32_t> &kernelSizesConv,
            std::vector<int32_t> &strideSizesConv, std::vector<int32_t> &dilationSizesConv, int32_t numActions,
            const std::optional<std::vector<int32_t>> &kernelSizesPool,
            const std::optional<std::vector<int32_t>> &strideSizesPool,
            const std::optional<std::vector<int32_t>> &dilationSizesPool,
            std::optional<std::shared_ptr<ActivationBase>> activation,
            float_t dropout, bool usePadding
    ) : model::ModelBase() {

        activationSubmodule = activation.has_value()
                              ? activation.value()
                              : std::make_shared<Relu>();

        usePadding_ = usePadding;
        usePooling_ = kernelSizesPool.has_value();

        setupModel(
                sequenceLength, channels, kernelSizesConv, strideSizesConv,
                dilationSizesConv, kernelSizesPool, strideSizesPool, dilationSizesPool,
                dropout, numActions);
    }

    Dcqn1d::Dcqn1d(std::unique_ptr<Dcqn1dOptions> &options) : model::ModelBase() {
        activationSubmodule = options->get_activation().has_value()
                              ? options->get_activation().value()
                              : std::make_shared<Relu>();

        usePadding_ = options->get_use_padding();
        usePooling_ = options->get_pool_kernel_sizes().has_value();

        auto sequenceLength = options->get_sequence_length();
        auto channels = options->get_channels();
        auto convKernelSizes = options->get_conv_kernel_sizes();
        auto convStridesSizes = options->get_conv_strides_sizes();
        auto convDilationSizes = options->get_conv_dilation_sizes();
        auto poolKernelSizes = options->get_pool_kernel_sizes();
        auto poolStridesSizes = options->get_pool_strides_sizes();
        auto poolDilationSizes = options->get_pool_dilation_sizes();
        auto dropout = options->get_dropout();
        auto numActions = options->get_num_actions();

        setupModel(sequenceLength, channels, convKernelSizes, convStridesSizes,
                   convDilationSizes, poolKernelSizes, poolStridesSizes,
                   poolDilationSizes, dropout, numActions);
    }

    Dcqn1d::~Dcqn1d() = default;

    std::vector<int32_t> Dcqn1d::get_interims(int32_t imageDims,
                                              std::vector<int32_t> &kernelSizesConv,
                                              std::vector<int32_t> &stridesSizesConv,
                                              std::vector<int32_t> &dilationSizesConv,
                                              const std::optional<std::vector<int32_t>> &kernelSizesPool,
                                              const std::optional<std::vector<int32_t>> &strideSizesPool,
                                              const std::optional<std::vector<int32_t>> &dilationSizesPool) const {
        size_t numBlocks = kernelSizesConv.size();
        std::vector<int32_t> interimSizes;
        interimSizes.push_back(imageDims);

        for (int32_t idx = 0; idx != numBlocks; idx++) {

            int32_t interimSizeConvLength = floor(
                    (((interimSizes[idx] - dilationSizesConv[idx] * (kernelSizesConv[idx] - 1) - 1) /
                      stridesSizesConv[idx])) + 1);

            if (usePooling_) {
                int32_t interimSizePoolLength = floor(
                        (((interimSizeConvLength -
                           dilationSizesPool.value()[idx] * (kernelSizesPool.value()[idx] - 1) -
                           1) / strideSizesPool.value()[idx])) + 1);
                interimSizes.push_back(interimSizePoolLength);
            } else {
                interimSizes.push_back(interimSizeConvLength);
            }
        }

        return interimSizes;
    }

    std::vector<std::vector<int32_t>> Dcqn1d::compute_padding(
            int32_t imageDims,
            std::vector<int32_t> &kernelSizesConv,
            std::vector<int32_t> &stridesSizesConv,
            std::vector<int32_t> &dilationSizesConv,
            const std::optional<std::vector<int32_t>> &kernelSizesPool,
            const std::optional<std::vector<int32_t>> &strideSizesPool,
            const std::optional<std::vector<int32_t>> &dilationSizesPool
    ) const {
        size_t numBlocks = kernelSizesConv.size();
        int32_t paddingConv, paddingPool;
        std::vector<int32_t> paddingsConv, paddingsPool;

        for (int32_t idx = 0; idx != numBlocks; idx++) {

            if (stridesSizesConv[idx] > 1) {
                throw std::runtime_error("Stride value greater than 1 are not supported when using padding!");
            }
            paddingConv = floor(
                    (imageDims * (stridesSizesConv[idx] - 1) +
                     dilationSizesConv[idx] * (kernelSizesConv[idx] - 1)) /
                    2);
            paddingsConv.push_back(paddingConv);

            if (usePooling_) {
                if (strideSizesPool.value()[idx] > 1) {
                    throw std::runtime_error("Stride value greater than 1 are not supported when using padding!");
                }
                paddingPool = floor((imageDims * (strideSizesPool.value()[idx] - 1) +
                                     dilationSizesPool.value()[idx] * (kernelSizesPool.value()[idx] - 1)) / 2);
                paddingsPool.push_back(paddingPool);
            }
        }

        std::vector<std::vector<int32_t>> paddings = {paddingsConv, paddingsPool};
        return paddings;
    }

    void Dcqn1d::setupModel(
            int32_t imageDims, std::vector<int32_t> &channels,
            std::vector<int32_t> &kernelSizesConv,
            std::vector<int32_t> &stridesSizesConv,
            std::vector<int32_t> &dilationSizesConv,
            const std::optional<std::vector<int32_t>> &kernelSizesPool,
            const std::optional<std::vector<int32_t>> &strideSizesPool,
            const std::optional<std::vector<int32_t>> &dilationSizesPool,
            float_t dropout, int32_t numClasses
    ) {

        size_t numBlocks = kernelSizesConv.size();
        std::vector<std::vector<int32_t>> paddings;

        std::vector<int32_t> interimSizes = get_interims(imageDims, kernelSizesConv, stridesSizesConv,
                                                         dilationSizesConv, kernelSizesPool, strideSizesPool,
                                                         dilationSizesPool);

        if (interimSizes.back() == 0) {
            throw std::runtime_error("The final output from convolution layers results in Tensor shape 0!");
        }

        if (usePadding_) {
            paddings = compute_padding(imageDims, kernelSizesConv, stridesSizesConv,
                                       dilationSizesConv, kernelSizesPool, strideSizesPool,
                                       dilationSizesPool);
        }

        std::string convModuleName = "conv_";
        std::string poolModuleName = "pool_";

        for (int32_t idx = 0; idx != numBlocks; idx++) {
            torch::nn::Conv1dOptions conv1dOptions = torch::nn::Conv1dOptions(channels[idx],
                                                                              channels[idx + 1],
                                                                              kernelSizesConv[idx])
                    .stride(stridesSizesConv[idx])
                    .dilation(dilationSizesConv[idx]);

            if (usePadding_) {
                conv1dOptions.padding(paddings.at(0)[idx]);
            }

            auto convBlock = std::make_shared<torch::nn::Conv1d>(conv1dOptions);
            convSubmodules->push_back(*convBlock);
            register_module(convModuleName.append((std::to_string(idx))), *convBlock);

            if (usePooling_) {
                torch::nn::MaxPool1dOptions maxPool1dOptions = torch::nn::MaxPool1dOptions(
                        kernelSizesPool.value()[idx])
                        .stride(strideSizesPool.value()[idx])
                        .dilation(dilationSizesPool.value()[idx]);

                if (usePadding_) {
                    maxPool1dOptions.padding(paddings.at(1)[idx]);
                }
                auto maxPool1d = std::make_shared<torch::nn::MaxPool1d>(maxPool1dOptions);
                maxPoolSubmodule->push_back(*maxPool1d);
                register_module(poolModuleName.append((std::to_string(idx))), *maxPool1d);
            }
        }

        int32_t finalSizes = imageDims;

        if (!usePadding_) {
            finalSizes = interimSizes.back();
        }

        auto dropoutOptions = torch::nn::DropoutOptions(dropout);
        dropoutSubmodule = std::make_shared<torch::nn::Dropout>(dropoutOptions);
        register_module("dropout", *dropoutSubmodule);

        auto flattenOptions = torch::nn::FlattenOptions().start_dim(1).end_dim(-1);
        flattenSubmodule = std::make_shared<torch::nn::Flatten>(flattenOptions);
        register_module("flatten", *flattenSubmodule);

        auto linearOptions = torch::nn::LinearOptions(finalSizes * channels.back(), numClasses);
        linearSubmodule = std::make_shared<torch::nn::Linear>(linearOptions);
        register_module("linear", *linearSubmodule);
    }

    torch::Tensor Dcqn1d::forward(torch::Tensor x) {
        for (int32_t idx = 0; idx != convSubmodules->size(); idx++) {
            x = convSubmodules[idx]->template as<torch::nn::Conv1d>()->forward(x);
            x = activationSubmodule->operator()(x);

            if (usePooling_) {
                x = maxPoolSubmodule[idx]->template as<torch::nn::MaxPool1d>()->forward(x);
            }
        }

        x = flattenSubmodule->ptr()->forward(x);
        x = dropoutSubmodule->ptr()->forward(x);
        x = linearSubmodule->ptr()->forward(x);

        return x;
    }

}//namespace model::dqn
