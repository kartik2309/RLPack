//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_DQN1D_DQN1D_H_
#define RLPACK_DQN_DQN1D_DQN1D_H_

#include <torch/torch.h>
#include "../DqnImpl.hpp"

namespace dqn {

template<class Activation = torch::nn::ReLU>
class Dqn1d : public DqnImpl {
  torch::nn::ModuleList convSubmodules = torch::nn::ModuleList();
  torch::nn::ModuleList maxPoolSubmodule = torch::nn::ModuleList();
  Activation activation;
  torch::nn::Flatten *flattenSubmodule = nullptr;
  torch::nn::Dropout *dropoutSubmodule = nullptr;
  torch::nn::Linear *linearSubmodule = nullptr;

  bool usePooling_;
  bool usePadding_;

  std::vector<int64_t> get_interims(int64_t imageDims,
                                           std::vector<int64_t> &kernelSizesConv,
                                           std::vector<int64_t> &stridesSizesConv,
                                           std::vector<int64_t> &dilationSizesConv,
                                           const std::optional<std::vector<int64_t>> &kernelSizesPool,
                                           const std::optional<std::vector<int64_t>> &strideSizesPool,
                                           const std::optional<std::vector<int64_t>> &dilationSizesPool);

  std::vector<std::vector<int64_t>> compute_padding(int64_t imageDims,
                                              std::vector<int64_t> &kernelSizesConv,
                                              std::vector<int64_t> &stridesSizesConv,
                                              std::vector<int64_t> &dilationSizesConv,
                                              const std::optional<std::vector<int64_t>> &kernelSizesPool,
                                              const std::optional<std::vector<int64_t>> &strideSizesPool,
                                              const std::optional<std::vector<int64_t>> &dilationSizesPool);

  void setupModel(int64_t &imageDims,
                  std::vector<int64_t> &channels,
                  std::vector<int64_t> &kernelSizesConv,
                  std::vector<int64_t> &stridesSizesConv,
                  std::vector<int64_t> &dilationSizesConv,
                  const std::optional<std::vector<int64_t>> &kernelSizesPool,
                  const std::optional<std::vector<int64_t>> &strideSizesPool,
                  const std::optional<std::vector<int64_t>> &dilationSizesPool,
                  float_t dropout,
                  int64_t numClasses);

 public:
  Dqn1d(
    int64_t sequenceLength,
    std::vector<int64_t> &channels,
    std::vector<int64_t> &kernelSizesConv,
    std::vector<int64_t> &strideSizesConv,
    std::vector<int64_t> &dilationSizesConv,
    int64_t numActions,
    const std::optional<std::vector<int64_t>> &kernelSizesPool = std::optional<std::vector<int64_t>>(),
    const std::optional<std::vector<int64_t>> &strideSizesPool = std::optional<std::vector<int64_t>>(),
    const std::optional<std::vector<int64_t>> &dilationSizesPool = std::optional<std::vector<int64_t>>(),
    std::optional<Activation> activation = std::optional<Activation>(),
    float_t dropout = 0.5,
    bool usePadding = true);

  ~Dqn1d() override;

  torch::Tensor forward(torch::Tensor x) override;
  void to_double() override;
};

}// namespace dqn

#endif//RLPACK_DQN_DQN1D_DQN1D_H_
