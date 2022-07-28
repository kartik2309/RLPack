//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_DCQN1D_DCQN1D_H_
#define RLPACK_DQN_DCQN1D_DCQN1D_H_

#include <torch/torch.h>
#include "Dcqn1dOptions/Dcqn1dOptions.h"
#include "../../utils/Base/ModelBase/ModelBase.h"
#include "../../Activations/Activation.hpp"

namespace model::dqn {

    class Dcqn1d : public model::ModelBase {
        torch::nn::ModuleList convSubmodules = torch::nn::ModuleList();
        torch::nn::ModuleList maxPoolSubmodule = torch::nn::ModuleList();
        std::shared_ptr<ActivationBase> activationSubmodule;
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
                                          const std::optional<std::vector<int32_t>> &dilationSizesPool) const;

        std::vector<std::vector<int32_t>> compute_padding(int32_t imageDims, std::vector<int32_t> &kernelSizesConv,
                                                          std::vector<int32_t> &stridesSizesConv,
                                                          std::vector<int32_t> &dilationSizesConv,
                                                          const std::optional<std::vector<int32_t>> &kernelSizesPool,
                                                          const std::optional<std::vector<int32_t>> &strideSizesPool,
                                                          const std::optional<std::vector<int32_t>> &dilationSizesPool) const;

        void setupModel(int32_t imageDims, std::vector<int32_t> &channels, std::vector<int32_t> &kernelSizesConv,
                        std::vector<int32_t> &stridesSizesConv, std::vector<int32_t> &dilationSizesConv,
                        const std::optional<std::vector<int32_t>> &kernelSizesPool,
                        const std::optional<std::vector<int32_t>> &strideSizesPool,
                        const std::optional<std::vector<int32_t>> &dilationSizesPool,
                        float_t dropout, int32_t numClasses);

    public:
        Dcqn1d(
                int32_t sequenceLength, std::vector<int32_t> &channels, std::vector<int32_t> &kernelSizesConv,
                std::vector<int32_t> &strideSizesConv, std::vector<int32_t> &dilationSizesConv, int32_t numActions,
                const std::optional<std::vector<int32_t>> &kernelSizesPool = std::optional<std::vector<int32_t>>(),
                const std::optional<std::vector<int32_t>> &strideSizesPool = std::optional<std::vector<int32_t>>(),
                const std::optional<std::vector<int32_t>> &dilationSizesPool = std::optional<std::vector<int32_t>>(),
                std::optional<std::shared_ptr<ActivationBase>> activation = std::optional<std::shared_ptr<ActivationBase>>(),
                float_t dropout = 0.5, bool usePadding = true
        );

        explicit Dcqn1d(std::unique_ptr<Dcqn1dOptions> &options);

        ~Dcqn1d() override;

        torch::Tensor forward(torch::Tensor x) override;
    };

}//namespace model::dqn

#endif//RLPACK_DQN_DCQN1D_DCQN1D_H_
