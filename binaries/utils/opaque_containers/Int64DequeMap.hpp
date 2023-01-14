//
// Created by Kartik Rajeshwaran on 2023-01-13.
//

#ifndef RLPACK_BINARIES_UTILS_INT64DEQUEMAP_HPP_
#define RLPACK_BINARIES_UTILS_INT64DEQUEMAP_HPP_

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
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::deque<int64_t>>)
void bind_int64_deque_map(pybind11::module& m) {
    /*!
     * Method to bind Int64DequeMap to any module as an opaque container.
     *
     * @param m: The pybind11 module scope.
     */
    pybind11::bind_map<std::map<std::string, std::deque<int64_t>>>(m, "Int64DequeMap")
            .def(
                    "__repr__",
                    [](std::map<std::string, std::deque<int64_t>> &int64DequeMap) {
                        std::string reprString;
                        std::stringstream ss;
                        ss << &int64DequeMap;
                        reprString = "<Int64DequeMap object at " + ss.str() + ">";
                        return reprString;
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "__str__",
                    [](std::map<std::string, std::deque<int64_t>> &int64DequeMap) {
                        std::string strString;
                        std::stringstream tensorString;
                        strString.append("{");
                        for (auto &pair: int64DequeMap) {
                            strString.append(pair.first);
                            strString.append(": [");
                            for (auto &value: pair.second) {
                                strString.append(std::to_string(value) + ", ");
                            }
                            strString.append("], ");
                        }
                        strString.append("}");
                        return strString;
                    },
                    pybind11::return_value_policy::reference)
            .def(pybind11::pickle(
                         [](std::map<std::string, std::deque<int64_t>> &int64DequeMap) {
                         pybind11::dict int64DequeMapDict;
                         for (auto &pair : int64DequeMap) {
                             int64DequeMapDict[pair.first.c_str()] = pair.second;
                         }
                         return int64DequeMapDict; },
                         [](pybind11::dict &init) {
                         std::map<std::string, std::deque<int64_t>> int64DequeMap;
                         for (auto &pair : init) {
                             int64DequeMap[pair.first.cast<std::string>()]
                                     = pair.second.cast<std::deque<int64_t>>();
                         }
                         return int64DequeMap; }),
                 pybind11::return_value_policy::reference);
}
/*!
 * @} @I{ // End opaque_containers_group }
 * @} @I{ // End utils_group }
 * @} @I{ // End binary_group }
 */

#endif//RLPACK_BINARIES_UTILS_INT64DEQUEMAP_HPP_
