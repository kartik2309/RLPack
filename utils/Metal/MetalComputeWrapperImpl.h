//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_UTILS_METALCOMPUTEWRAPPER_H_
#define RLPACK_UTILS_METALCOMPUTEWRAPPER_H_

#include <Metal/Metal.hpp>
#include <iostream>

class MetalComputeWrapperImpl {

 protected:
  virtual void encodeComputeCommand(MTL::ComputeCommandEncoder *computeEncoder) = 0;
  virtual void prepareData() = 0;

 public:
  MTL::Device *metalDevice{};
  MTL::ComputePipelineState *metalComputeFunctionPso{};
  MTL::CommandQueue *metalCommandQueue{};

  void initialize(MTL::Device *);
  void execute_compute_command();
};

#endif//RLPACK_UTILS_METALCOMPUTEWRAPPER_H_
