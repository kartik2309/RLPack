//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#include "GetDqnAgent.h"

PYBIND11_MODULE(RLPack, m) {
    m.doc() = "RLPack plugin to bind the Dqn set of algorithms to Python interface";
    pybind11::class_<GetDqnAgent>(m, "GetDqnAgent")
            .def(pybind11::init<pybind11::str &, pybind11::dict &, pybind11::dict &,
                    pybind11::dict &, pybind11::dict &, pybind11::dict &, pybind11::str &>())
            .def("train", &GetDqnAgent::train, "train method to train the agent.")
            .def("policy", &GetDqnAgent::policy, "policy method to run the policy (only eval) of the agent.")
            .def("setup_agent", &GetDqnAgent::setup_agent,
                 "setup method to allocate memory and setup the agent and all underlying modules.")
            .def("save", &GetDqnAgent::save, "save method to save the policy and target model of the agent.")
            .def("load", &GetDqnAgent::load, "load method to save the policy and target model of the agent.")
            .def("__repr__", [](const GetDqnAgent &) { return "<GetDqnAgent>"; });
}
