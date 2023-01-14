//
// Created by Kartik Rajeshwaran on 2022-12-14.
//

#include "C_GradAccumulator.h"
#include "opaque_containers/TensorMap.hpp"

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup grad_accumulator_group grad_accumulator
 * @brief Grad accumulator module is the C++ backend for rlpack._C.grad_accumulator.GradAccumulator class.
 * @{
 */

PYBIND11_MODULE(C_GradAccumulator, m) {
    /*!
     * Python bindings for C_GradAccumulator. Only relevant public methods are exposed to Python.
     */
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
            .def("mean_reduce",
                 &C_GradAccumulator::mean_reduce,
                 "Method to perform mean reduction for named parameters' gradients",
                 pybind11::return_value_policy::reference)
            .def("sum_reduce",
                 &C_GradAccumulator::sum_reduce,
                 "Method to perform sum reduction for named parameters' gradients",
                 pybind11::return_value_policy::reference)
            .def("get_item",
                 &C_GradAccumulator::get_item,
                 "Method to get named parameter gradients at a given index.",
                 pybind11::return_value_policy::reference,
                 pybind11::arg("index"))
            .def("set_item",
                 &C_GradAccumulator::set_item,
                 "Method to set named parameter gradients at a given index.",
                 pybind11::arg("index"),
                 pybind11::arg("named_parameters"))
            .def("delete_item",
                 &C_GradAccumulator::delete_item,
                 "Method to delete named parameter gradients at a given index.",
                 pybind11::arg("index"))
            .def("size",
                 &C_GradAccumulator::size,
                 "Method to retrieve the size of the C_GradAccumulator i.e. the number of accumulated parameters")
            .def("clear", &C_GradAccumulator::clear, "Method to clear accumulated gradients.");
    // Bind TensorMap
    bind_tensor_map(m);
}
/*!
 * @} @I{ // End group grad_accumulator_group }
 * @} @I{ // End group binaries_group }
 */
