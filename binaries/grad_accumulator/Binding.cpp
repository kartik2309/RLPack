//
// Created by Kartik Rajeshwaran on 2022-12-14.
//

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "C_GradAccumulator.h"

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup grad_accumulator_group grad_accumulator
 * @brief Memory module is the C++ backend for rlpack._C.grad_accumulator.GradAccumulator class. Heavier workloads
 * have been optimized with multithreading with OpenMP.
 * @{
 */
PYBIND11_MAKE_OPAQUE(std::map<std::string, torch::Tensor>)
PYBIND11_MODULE(C_GradAccumulator, m) {

    m.doc() = "Module to provide Python binding for C_GradAccumulator class";
    pybind11::class_<C_GradAccumulator>(m, "C_GradAccumulator")
            .def(pybind11::init<std::vector<std::string> &, int64_t>(),
                 "Class constructor for C_GradAccumulator.",
                 pybind11::arg("parameter_keys"),
                 pybind11::arg("bootstrap_rounds"))
            .def("accumulate",
                 &C_GradAccumulator::accumulate,
                 "Method to accumulate gradients.",
                 pybind11::arg("named_parameters"))
            .def("mean_reduce", &C_GradAccumulator::mean_reduce, "Method to perform mean reduction.")
            .def("clear", &C_GradAccumulator::clear, "Method to clear accumulated gradients.");

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
                     return mapOfTensors; }),
                 "Pickle method for mapOfTensors.", pybind11::return_value_policy::reference);
}
/*!
 * @} @I{ // End group grad_accumulator_group }
 * @} @I{ // End group binaries_group }
 */
