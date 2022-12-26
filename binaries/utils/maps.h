//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#ifndef RLPACK_BINARIES_UTILS_MAPS_H_
#define RLPACK_BINARIES_UTILS_MAPS_H_

#include <torch/extension.h>

class Maps {
public:
    inline static std::map<std::string, torch::DeviceType> deviceMap{
            {"cpu", torch::kCPU},
            {"cuda", torch::kCUDA},
            {"mps", torch::kMPS}};
    inline static std::map<std::string, torch::Dtype> dTypeMap{
            {"float32", torch::kFloat32},
            {"float64", torch::kFloat64},
    };
};

#endif//RLPACK_BINARIES_UTILS_MAPS_H_
