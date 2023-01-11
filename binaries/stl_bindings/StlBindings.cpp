//
// Created by Kartik Rajeshwaran on 2023-01-05.
//

#include "StlBindings.h"

PYBIND11_MODULE(StlBindings, m) {
    /*
     * Binding the opaque object std::map<std::string, std::deque<torch::Tensor>> to Python.
     * This will be exposed as TensorMap to Python.
     */
    pybind11::bind_map<std::map<std::string, torch::Tensor>>(m, "TensorMap")
            .def(
                    "__repr__",
                    [](std::map<std::string, torch::Tensor> &tensorMap) {
                        std::string reprString;
                        std::stringstream ss;
                        ss << &tensorMap;
                        reprString = "<TensorMap object at " + ss.str() + ">";
                        return reprString;
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "__str__",
                    [](std::map<std::string, torch::Tensor> &tensorMap) {
                        std::string strString;
                        std::stringstream tensorString;
                        strString.append("{");
                        for (auto &pair: tensorMap) {
                            tensorString << pair.second;
                            strString.append(pair.first + ":" + tensorString.str() + " ");
                            tensorString.clear();
                        }
                        strString.append("}");
                        return strString;
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    pybind11::pickle(
                            [](std::map<std::string, torch::Tensor> &tensorMap) {
                                pybind11::dict tensorMapDict;
                                for (auto &pair : tensorMap) {
                                    tensorMapDict[pair.first.c_str()] = pair.second;
                                }
                                return tensorMapDict; },
                            [](pybind11::dict &init) {
                                std::map<std::string, torch::Tensor> tensorMap;
                                for (auto &pair : init) {
                                    tensorMap[pair.first.cast<std::string>()] = pair.second.cast<torch::Tensor>();
                                }
                                return tensorMap; }),
                    "Pickle method for TensorMap.",
                    pybind11::return_value_policy::reference);

    pybind11::bind_map<std::map<std::string, c10::intrusive_ptr<c10d::ProcessGroup>>>(m, "ProcessGroupMap");

    /*
     * Binding the opaque object std::map<std::string, std::deque<torch::Tensor>> to Python.
     * This will be exposed as TensorDequeMap to Python.
     */
    pybind11::bind_map<std::map<std::string, std::deque<torch::Tensor>>>(m, "TensorDequeMap")
            .def(
                    "__repr__",
                    [](std::map<std::string, std::deque<torch::Tensor>> &tensorDequeMap) {
                        std::string reprString;
                        std::stringstream ss;
                        ss << &tensorDequeMap;
                        reprString = "<TensorDequeMap object at " + ss.str() + ">";
                        return reprString;
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "__str__",
                    [](std::map<std::string, std::deque<torch::Tensor>> &tensorDequeMap) {
                        std::string strString;
                        std::stringstream tensorString;
                        strString.append("{");
                        for (auto &pair: tensorDequeMap) {
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
                         [](std::map<std::string, std::deque<torch::Tensor>> &tensorDequeMap) {
                             pybind11::dict tensorDequeMapDict;
                             for (auto &pair : tensorDequeMap) {
                                 tensorDequeMapDict[pair.first.c_str()] = pair.second;
                             }
                             return tensorDequeMapDict; },
                         [](pybind11::dict &init) {
                             std::map<std::string, std::deque<torch::Tensor>> tensorDequeMap;
                             for (auto &pair : init) {
                                 tensorDequeMap[pair.first.cast<std::string>()] = pair.second.cast<std::deque<torch::Tensor>>();
                             }
                             return tensorDequeMap; }),
                 pybind11::return_value_policy::reference);
    /*
     * Binding the opaque object std::map<std::string, std::deque<int64_t>> to Python.
     * This will be Int64DequeMap as int64DequeMap to Python.
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
                                 int64DequeMap[pair.first.cast<std::string>()] = pair.second.cast<std::deque<int64_t>>();
                             }
                             return int64DequeMap; }),
                 pybind11::return_value_policy::reference);
}