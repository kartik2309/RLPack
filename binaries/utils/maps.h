//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#ifndef RLPACK_BINARIES_UTILS_MAPS_H_
#define RLPACK_BINARIES_UTILS_MAPS_H_

#include <torch/extension.h>

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup utils_group utils
 * @brief Utils Module provides generic utilities to be used by all binaries in rlpack.
 * @{
 */
 /*!
  * @brief Maps class provides generic utility mappings for strings with various objects.
  */
class Maps {
public:
    //! The map between std::string and torch::DeviceType; mapping the device name in string to DeviceType.
    inline static std::map<std::string, torch::DeviceType> deviceMap{
            {"cpu", torch::kCPU},
            {"cuda", torch::kCUDA},
            {"mps", torch::kMPS}};
    //! The map between std::string and torch::Dtype; mapping the device name in string to DataType.
    inline static std::map<std::string, torch::Dtype> dTypeMap{
            {"float32", torch::kFloat32},
            {"float64", torch::kFloat64},
    };
};
/*!
 * @} @I{ // End utils_group }
 * @} @I{ // End binary_group }
 */

#endif//RLPACK_BINARIES_UTILS_MAPS_H_
