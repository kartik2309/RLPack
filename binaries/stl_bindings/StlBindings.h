//
// Created by Kartik Rajeshwaran on 2023-01-05.
//

#ifndef RLPACK_BINARIES_UTILS_STL_BINDINGS_STLBINDINGS_H_
#define RLPACK_BINARIES_UTILS_STL_BINDINGS_STLBINDINGS_H_

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

PYBIND11_MAKE_OPAQUE(std::map<std::string, torch::Tensor>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, c10::intrusive_ptr<c10d::ProcessGroup>>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::deque<int64_t>>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::deque<torch::Tensor>>)


#endif//RLPACK_BINARIES_UTILS_STL_BINDINGS_STLBINDINGS_H_
