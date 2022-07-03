//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_DCQN1D_DCQN1D_H_
#define RLPACK_DQN_DCQN1D_DCQN1D_H_

#include <torch/torch.h>
#include "../DqnBase.hpp"

namespace dqn {

    template<class Activation = torch::nn::ReLU>
    class Dcqn1d : public DqnBase {
        torch::nn::ModuleList  convSubmodules = torch::nn::ModuleList();
        torch::nn::ModuleList maxPoolSubmodule = torch::nn::ModuleList();
        Activation activationSubmodule;
        std::shared_ptr<torch::nn::Flatten> flattenSubmodule = nullptr;
        std::shared_ptr<torch::nn::Dropout> dropoutSubmodule = nullptr;
        std::shared_ptr<torch::nn::Linear> linearSubmodule = nullptr;

        bool usePooling_;
        bool usePadding_;

        std::vector<int32_t> get_interims(int32_t imageDims, std::vector<int32_t> &kernelSizesConv,
                                          std::vector<int32_t> &stridesSizesConv,
                                          std::vector<int32_t> &dilationSizesConv,
                                          const std::optional<std::vector<int32_t>> &kernelSizesPool,
                                          const std::optional<std::vector<int32_t>> &strideSizesPool,
                                          const std::optional<std::vector<int32_t>> &dilationSizesPool);

        std::vector<std::vector<int32_t>> compute_padding(int32_t imageDims, std::vector<int32_t> &kernelSizesConv,
                                                          std::vector<int32_t> &stridesSizesConv,
                                                          std::vector<int32_t> &dilationSizesConv,
                                                          const std::optional<std::vector<int32_t>> &kernelSizesPool,
                                                          const std::optional<std::vector<int32_t>> &strideSizesPool,
                                                          const std::optional<std::vector<int32_t>> &dilationSizesPool);

        void setupModel(int32_t &imageDims, std::vector<int32_t> &channels, std::vector<int32_t> &kernelSizesConv,
                        std::vector<int32_t> &stridesSizesConv, std::vector<int32_t> &dilationSizesConv,
                        const std::optional<std::vector<int32_t>> &kernelSizesPool,
                        const std::optional<std::vector<int32_t>> &strideSizesPool,
                        const std::optional<std::vector<int32_t>> &dilationSizesPool,
                        float_t dropout, int32_t numClasses);

    public:
        Dcqn1d(int32_t sequenceLength, std::vector<int32_t> &channels, std::vector<int32_t> &kernelSizesConv,
               std::vector<int32_t> &strideSizesConv, std::vector<int32_t> &dilationSizesConv, int32_t numActions,
               const std::optional<std::vector<int32_t>> &kernelSizesPool = std::optional<std::vector<int32_t>>(),
               const std::optional<std::vector<int32_t>> &strideSizesPool = std::optional<std::vector<int32_t>>(),
               const std::optional<std::vector<int32_t>> &dilationSizesPool = std::optional<std::vector<int32_t>>(),
               std::optional<Activation> activation = std::optional<Activation>(), float_t dropout = 0.5,
               bool usePadding = true);

        ~Dcqn1d() override;

        torch::Tensor forward(torch::Tensor x) override;
    };

}// namespace dqn

#endif//RLPACK_DQN_DCQN1D_DCQN1D_H_
