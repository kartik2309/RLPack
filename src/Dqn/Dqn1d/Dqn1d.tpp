//
// Created by Kartik Rajeshwaran on 2022-06-19.
//

#ifndef RLPACK_DQN_DQN1D_DQN1D_TPP_
#define RLPACK_DQN_DQN1D_DQN1D_TPP_

#include "Dqn1d.h"

namespace dqn {

template<class Activation>
Dqn1d<Activation>::Dqn1d(
  int64_t sequenceLength,
  std::vector<int64_t> &channels,
  std::vector<int64_t> &kernelSizesConv,
  std::vector<int64_t> &strideSizesConv,
  std::vector<int64_t> &dilationSizesConv,
  int64_t numActions,
  const std::optional<std::vector<int64_t>> &kernelSizesPool,
  const std::optional<std::vector<int64_t>> &strideSizesPool,
  const std::optional<std::vector<int64_t>> &dilationSizesPool,
  std::optional<Activation> activation,
  float_t dropout,
  bool usePadding) {

  usePadding_ = usePadding;
  usePooling_ = kernelSizesPool.has_value();

  setupModel(sequenceLength,
             channels,
             kernelSizesConv,
             strideSizesConv,
             dilationSizesConv,
             kernelSizesPool,
             strideSizesPool,
             dilationSizesPool,
             dropout,
             numActions);
}

template<class Activation>
Dqn1d<Activation>::~Dqn1d<Activation>() = default;

template<class Activation>
std::vector<int64_t> Dqn1d<Activation>::get_interims(int64_t imageDims,
                                                     std::vector<int64_t> &kernelSizesConv,
                                                     std::vector<int64_t> &stridesSizesConv,
                                                     std::vector<int64_t> &dilationSizesConv,
                                                     const std::optional<std::vector<int64_t>> &kernelSizesPool,
                                                     const std::optional<std::vector<int64_t>> &strideSizesPool,
                                                     const std::optional<std::vector<int64_t>> &dilationSizesPool) {
  size_t numBlocks = kernelSizesConv.size();
  std::vector<int64_t> interimSizes;
  interimSizes.push_back(imageDims);

  for (int64_t idx = 0; idx != numBlocks; idx++) {

    int64_t interimSizeConvLength = floor(
      (((interimSizes[idx] - dilationSizesConv[idx] * (kernelSizesConv[idx] - 1) - 1) / stridesSizesConv[idx])) + 1);

    if (usePooling_) {
      int64_t interimSizePoolLength = floor(
        (((interimSizeConvLength - dilationSizesPool.value()[idx] * (kernelSizesPool.value()[idx] - 1) - 1) / strideSizesPool.value()[idx])) + 1);
      interimSizes.push_back(interimSizePoolLength);
    } else {
      interimSizes.push_back(interimSizeConvLength);
    }
  }

  return interimSizes;
}

template<class Activation>
std::vector<std::vector<int64_t>> Dqn1d<Activation>::compute_padding(int64_t imageDims,
                                                                     std::vector<int64_t> &kernelSizesConv,
                                                                     std::vector<int64_t> &stridesSizesConv,
                                                                     std::vector<int64_t> &dilationSizesConv,
                                                                     const std::optional<std::vector<int64_t>> &kernelSizesPool,
                                                                     const std::optional<std::vector<int64_t>> &strideSizesPool,
                                                                     const std::optional<std::vector<int64_t>> &dilationSizesPool) {
  size_t numBlocks = kernelSizesConv.size();
  int64_t paddingConv, paddingPool;
  std::vector<int64_t> paddingsConv, paddingsPool;

  for (int64_t idx = 0; idx != numBlocks; idx++) {

    if (stridesSizesConv[idx] > 1 and usePadding_) {
      throw std::runtime_error("Stride value greater than 1 are not supported when using padding!");
    }
    paddingConv = floor((imageDims * (stridesSizesConv[idx] - 1) + dilationSizesConv[idx] * (kernelSizesConv[idx] - 1)) / 2);
    paddingsConv.push_back(paddingConv);

    if (usePooling_) {
      paddingPool = floor((imageDims * (stridesSizesConv[idx] - 1) + dilationSizesConv[idx] * (kernelSizesConv[idx] - 1)) / 2);
      paddingsPool.push_back(paddingPool);
    }
  }

  std::vector<std::vector<int64_t>> paddings = {paddingsConv, paddingsPool};
  return paddings;
}

template<class Activation>
void Dqn1d<Activation>::setupModel(int64_t &imageDims,
                                   std::vector<int64_t> &channels,
                                   std::vector<int64_t> &kernelSizesConv,
                                   std::vector<int64_t> &stridesSizesConv,
                                   std::vector<int64_t> &dilationSizesConv,
                                   const std::optional<std::vector<int64_t>> &kernelSizesPool,
                                   const std::optional<std::vector<int64_t>> &strideSizesPool,
                                   const std::optional<std::vector<int64_t>> &dilationSizesPool,
                                   float_t dropout,
                                   int64_t numClasses) {

  size_t numBlocks = kernelSizesConv.size();
  std::vector<std::vector<int64_t>> paddings;

  std::vector<int64_t> interimSizes = get_interims(imageDims,
                                                   kernelSizesConv,
                                                   stridesSizesConv,
                                                   dilationSizesConv,
                                                   kernelSizesPool,
                                                   strideSizesPool,
                                                   dilationSizesPool);

  if (usePadding_) {
    paddings = compute_padding(imageDims,
                               kernelSizesConv,
                               stridesSizesConv,
                               dilationSizesConv,
                               kernelSizesPool,
                               strideSizesPool,
                               dilationSizesPool);
  }

  std::string convModuleName = "conv_";
  std::string poolModuleName = "pool_";

  for (int64_t idx = 0; idx != numBlocks; idx++) {
    torch::nn::Conv1dOptions conv1dOptions = torch::nn::Conv1dOptions(channels[idx],
                                                                      channels[idx + 1],
                                                                      kernelSizesConv[idx])
                                               .stride(stridesSizesConv[idx])
                                               .dilation(dilationSizesConv[idx]);

    if (usePadding_ and !paddings.at(0).empty()) {
      conv1dOptions.padding(paddings.at(0)[idx]);
    }

    auto *convBlock = new torch::nn::Conv1d(conv1dOptions);
    convSubmodules->push_back(*convBlock);
    register_module(convModuleName.append((std::to_string(idx))), *convBlock);

    if (usePooling_) {
      torch::nn::MaxPool1dOptions maxPool1dOptions = torch::nn::MaxPool1dOptions(kernelSizesPool.value()[idx])
                                                       .stride(strideSizesPool.value()[idx])
                                                       .dilation(dilationSizesPool.value()[idx]);

      if (usePadding_) {
        maxPool1dOptions.padding(paddings.at(1)[idx]);
      }
      auto *maxPool1d = new torch::nn::MaxPool1d(maxPool1dOptions);
      maxPoolSubmodule->push_back(*maxPool1d);
      register_module(poolModuleName.append((std::to_string(idx))), *maxPool1d);
    }
  }

  int64_t finalSizes = imageDims;

  if (!usePadding_) {
    finalSizes = interimSizes.back();
  }

  auto dropoutOptions = torch::nn::DropoutOptions(dropout);
  dropoutSubmodule = new torch::nn::Dropout(dropoutOptions);
  register_module("dropout", *dropoutSubmodule);

  auto flattenOptions = torch::nn::FlattenOptions().start_dim(1).end_dim(-1);
  flattenSubmodule = new torch::nn::Flatten(flattenOptions);
  register_module("flatten", *flattenSubmodule);

  auto linearOptions = torch::nn::LinearOptions(finalSizes * channels.back(), numClasses);
  linearSubmodule = new torch::nn::Linear(linearOptions);
  register_module("linear", *linearSubmodule);
}

template<class Activation>
torch::Tensor Dqn1d<Activation>::forward(torch::Tensor x) {
  for (int64_t idx = 0; idx != convSubmodules->size(); idx++) {
    x = convSubmodules[idx]->template as<torch::nn::Conv1d>()->forward(x);
    x = activation(x);

    if (usePooling_) {
      x = maxPoolSubmodule[idx]->template as<torch::nn::MaxPool1d>()->forward(x);
    }
  }

  x = flattenSubmodule->ptr()->forward(x);
  x = dropoutSubmodule->ptr()->forward(x);
  x = linearSubmodule->ptr()->forward(x);
  return x;
}

template<class Activation>
void Dqn1d<Activation>::to_double() {

  for (int64_t idx = 0; idx != convSubmodules->size(); idx++) {
    convSubmodules[idx]->template as<torch::nn::Conv1d>()->weight = convSubmodules[idx]->template as<torch::nn::Conv1d>()->weight.toType(torch::kDouble);
    convSubmodules[idx]->template as<torch::nn::Conv1d>()->bias = convSubmodules[idx]->template as<torch::nn::Conv1d>()->bias.toType(torch::kDouble);
  }
  linearSubmodule->ptr()->bias = linearSubmodule->ptr()->bias.toType(torch::kDouble);
  linearSubmodule->ptr()->weight = linearSubmodule->ptr()->weight.toType(torch::kDouble);
}
}// namespace dqn

#endif//RLPACK_DQN_DQN1D_DQN1D_TPP_