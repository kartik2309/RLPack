//
// Created by Kartik Rajeshwaran on 2023-01-13.
//

#ifndef RLPACK_BINARIES_UTILS_TENSORVECTORMAP_HPP_
#define RLPACK_BINARIES_UTILS_TENSORVECTORMAP_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup utils_group utils
 * @brief Utils Module provides generic utilities to be used by all binaries in rlpack.
 * @{
 * @addtogroup opaque_containers_group opaque_containers
 * @brief opaque_containers module provides functions to make useful containers opaque and bind it with any
 * module. Opaque containers will not be casted by PyBind11 and hence are safe for moving tensors between C++ and
 * Python.
 * @{
 */
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::vector<torch::Tensor>>)
void bind_tensor_vector_map(pybind11::module& m){
    /*!
     * Method to bind TensorVectorMap to any module as an opaque container.
     *
     * @param m: The pybind11 module scope.
     */
    pybind11::bind_map<std::map<std::string, std::vector<torch::Tensor>>>(m, "TensorVectorMap")
            .def(
                    "__repr__",
                    [](std::map<std::string, std::vector<torch::Tensor>> &tensorVectorMap) {
                        std::string reprString;
                        std::stringstream ss;
                        ss << &tensorVectorMap;
                        reprString = "<TensorVectorMap object at " + ss.str() + ">";
                        return reprString;
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "__str__",
                    [](std::map<std::string, std::vector<torch::Tensor>> &tensorVectorMap) {
                        std::string strString;
                        std::stringstream tensorString;
                        strString.append("{");
                        for (auto &pair: tensorVectorMap) {
                            strString.append(pair.first);
                            strString.append(": [");
                            for (auto &tensor: pair.second) {
                                tensorString << tensor;
                                strString.append(tensorString.str() + ", ");
                            }
                            strString.append("], ");
                        }
                        strString.append("}");
                        return strString;
                    },
                    pybind11::return_value_policy::reference)
            .def(pybind11::pickle(
                         [](std::map<std::string, std::vector<torch::Tensor>> &tensorVectorMap) {
                             pybind11::dict tensorVectorMapDict;
                             for (auto &pair : tensorVectorMap) {
                                 tensorVectorMapDict[pair.first.c_str()] = pair.second;
                             }
                             return tensorVectorMapDict; },
                         [](pybind11::dict &init) {
                             std::map<std::string, std::vector<torch::Tensor>> tensorVectorMap;
                             for (auto &pair : init) {
                                 tensorVectorMap[pair.first.cast<std::string>()]
                                         = pair.second.cast<std::vector<torch::Tensor>>();
                             }
                             return tensorVectorMap; }),
                 pybind11::return_value_policy::reference);
}
/*!
 * @} @I{ // End opaque_containers_group }
 * @} @I{ // End utils_group }
 * @} @I{ // End binary_group }
 */

#endif//RLPACK_BINARIES_UTILS_TENSORVECTORMAP_HPP_
