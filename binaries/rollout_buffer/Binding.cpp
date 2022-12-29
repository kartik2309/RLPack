//
// Created by Kartik Rajeshwaran on 2022-12-26.
//

#include <pybind11/stl_bind.h>

#include "C_RolloutBuffer.h"

PYBIND11_MAKE_OPAQUE(std::map<std::string, torch::Tensor>)
PYBIND11_MODULE(C_RolloutBuffer, m) {
    m.doc() = "Module to provide Python binding for RolloutBuffer class";
    pybind11::class_<C_RolloutBuffer>(m, "C_RolloutBuffer")
            .def(pybind11::init<int64_t, std::string &, std::string &>(),
                 "Class constructor for C_RolloutBuffer.",
                 pybind11::arg("buffer_size"),
                 pybind11::arg("device"),
                 pybind11::arg("dtype"))
            .def("insert",
                 &C_RolloutBuffer::insert,
                 pybind11::arg("input_map"))
            .def("compute_returns",
                 &C_RolloutBuffer::compute_returns,
                 pybind11::arg("gamma"),
                 pybind11::return_value_policy::reference)
            .def("get_stacked_rewards",
                 &C_RolloutBuffer::get_stacked_rewards,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_action_log_probabilities",
                 &C_RolloutBuffer::get_stacked_action_log_probabilities,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_state_current_values",
                 &C_RolloutBuffer::get_stacked_state_current_values,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_entropies",
                 &C_RolloutBuffer::get_stacked_entropies,
                 pybind11::return_value_policy::reference)
            .def("clear",
                 &C_RolloutBuffer::clear)
            .def("size",
                 &C_RolloutBuffer::size);

    /*
     * Binding the opaque object std::map<std::string, torch::Tensor> to Python.
     * This will be exposed as MapOfTensors to Python.
     */
    pybind11::bind_map<std::map<std::string, torch::Tensor>>(m, "MapOfTensors")
            .def(
                    "__repr__", [](std::map<std::string, torch::Tensor> &mapOfTensors) {
                        std::string reprString;
                        std::stringstream ss;
                        ss << &mapOfTensors;
                        reprString = "<MapOfTensors object at " + ss.str() + ">";
                        return reprString;
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "__str__", [](std::map<std::string, torch::Tensor> &mapOfTensors) {
                        std::string strString;
                        std::stringstream tensorString;
                        strString.append("{\n");
                        for (auto &pair: mapOfTensors) {
                            tensorString << pair.second;
                            strString.append("\t" + pair.first + ":" + tensorString.str() + "\n");
                            tensorString.clear();
                        }
                        strString.append("}");
                        return strString;
                    },
                    pybind11::return_value_policy::reference)
            .def(pybind11::pickle([](std::map<std::string, torch::Tensor> &mapOfTensors) {
                     pybind11::dict mapOfTensorsDict;
                     for (auto &pair : mapOfTensors) {
                         mapOfTensorsDict[pair.first.c_str()] = pair.second;
                     }
                     return mapOfTensorsDict; }, [](pybind11::dict &init) {
                     std::map<std::string, torch::Tensor> mapOfTensors;
                     for (auto &pair : init) {
                         mapOfTensors[pair.first.cast<std::string>()] = pair.second.cast<torch::Tensor>();
                     }
                     return mapOfTensors; }), "Pickle method for MapOfTensors.", pybind11::return_value_policy::reference);
}