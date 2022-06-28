//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#include "MetalComputeWrapperImpl.h"

#include "MetalComputeWrapperImpl.h"
#include <iostream>

void MetalComputeWrapperImpl::initialize(MTL::Device *device) {
  metalDevice = device;
  NS::Error *error;

  auto defaultLibrary = metalDevice->newDefaultLibrary();

  if (!defaultLibrary) {
    throw std::runtime_error("Metal Library was not found!");
  }

  auto functionName = NS::String::string("add_arrays_in_metal", NS::ASCIIStringEncoding);
  auto computeFunction = defaultLibrary->newFunction(functionName);

  if (!computeFunction) {
    throw std::runtime_error("Invalid Compute Function received!");
  }

  metalComputeFunctionPso = metalDevice->newComputePipelineState(computeFunction, &error);
  metalCommandQueue = metalDevice->newCommandQueue();

  if (!metalCommandQueue) {
    throw std::runtime_error("Invalid Command Queue received was not found!");
  }
}

void MetalComputeWrapperImpl::execute_compute_command() {

  MTL::CommandBuffer *commandBuffer = metalCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);

  MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);

  encodeComputeCommand(computeEncoder);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}
