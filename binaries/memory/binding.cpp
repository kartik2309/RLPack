//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#include "C_Memory.h"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

PYBIND11_MAKE_OPAQUE(std::map<std::string, torch::Tensor>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::vector<torch::Tensor>>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::vector<int64_t>>)
PYBIND11_MODULE(C_Memory, m) {
  m.doc() = "Module to provide Python binding for C_Memory class";
  /*
  * This is the binding of class C_Memory aliased C_Memory. This is the main interface to connect with
  * C++ backend. This class exposes all the necessary functions with pickling and copy support.
  */
  pybind11::class_<C_Memory>(m, "C_Memory")
      .def(pybind11::init<pybind11::int_ &, pybind11::str &, pybind11::int_ &>(),
           pybind11::arg("buffer_size"),
           pybind11::arg("device"),
           pybind11::arg("parallelism_size_threshold"))
      .def("insert", &C_Memory::insert, "Insertion method to memory.",
           pybind11::arg("state_current"),
           pybind11::arg("state_next"),
           pybind11::arg("reward"),
           pybind11::arg("action"),
           pybind11::arg("done"),
           pybind11::arg("is_terminal"))
      .def("get_item", &C_Memory::get_item, "Get item method to get an item as per index.",
           pybind11::return_value_policy::reference,
           pybind11::arg("index"))
      .def("delete_item", &C_Memory::delete_item, "Delete item method to delete an item as per index.",
           pybind11::arg("index"))
      .def("set_item", &C_Memory::set_item, "Set item method to set item as per index.",
           pybind11::arg("index"),
           pybind11::arg("state_current"),
           pybind11::arg("state_next"),
           pybind11::arg("reward"),
           pybind11::arg("action"),
           pybind11::arg("done"),
           pybind11::arg("is_terminal"))
      .def("sample", &C_Memory::sample,
           "Sample items from memory."
           " Overload for when we pass both batchSize and forceTerminalStateProbability."
           " This method samples items and arranges them quantity-wise.",
           pybind11::return_value_policy::reference,
           pybind11::arg("batch_size"),
           pybind11::arg("force_terminal_state_probability"))
      .def("initialize", &C_Memory::initialize,
           "Initialize the Memory with input vector of values.",
           pybind11::arg("c_memory_data"))
      .def("clear", &C_Memory::clear, "Clear all items in memory.")
      .def("size", &C_Memory::size, "Return the current_size of the memory.",
           pybind11::return_value_policy::reference)
      .def("view", &C_Memory::view, "Return the current memory view.",
           pybind11::return_value_policy::reference)
      .def("__repr__", [](C_Memory &cMemory) {
             std::string reprString;
             std::stringstream ss;
             ss << &cMemory;
             reprString = "<C_Memory object at " + ss.str() + ">";
             return reprString;
           },
           pybind11::return_value_policy::reference)
      .def(pybind11::pickle(
               [](C_Memory &cMemory) { return cMemory.view(); },
               [](C_Memory::C_MemoryData &init) {
                 C_Memory cMemory;
                 cMemory.initialize(init);
                 return cMemory;
               }),
           "Pickle method for C_Memory.",
           pybind11::return_value_policy::reference
      );

  /*
   * This is the binding of class C_MemoryData aliased C_MemoryData.
   */
  pybind11::class_<C_Memory::C_MemoryData>(m, "C_MemoryData")
      .def("__repr__", [](C_Memory::C_MemoryData &cMemoryData) {
        std::string reprString;
        std::stringstream ss;
        ss << &cMemoryData;
        reprString = "<C_Memory::C_MemoryData object at " + ss.str() + ">";
        return reprString;
      })
      .def(pybind11::pickle(
               [](C_Memory::C_MemoryData &cMemoryData) {
                 pybind11::dict cMemoryDataDict;
                 cMemoryDataDict["transition_information"] = cMemoryData.deref_transition_information_map();
                 cMemoryDataDict["terminal_state_indices"] = cMemoryData.deref_terminal_state_indices();
                 cMemoryDataDict["buffer_size"] = cMemoryData.get_buffer_size();
                 cMemoryDataDict["parallelism_size_threshold"] = cMemoryData.get_parallelism_size_threshold();
                 return cMemoryDataDict;
               },
               [](pybind11::dict &init) {
                 auto bufferSize = init["buffer_size"].cast<int64_t>();
                 auto parallelismSizeThreshold = init["parallelism_size_threshold"].cast<int64_t>();
                 C_Memory::C_MemoryData cMemoryData(&bufferSize,
                                                    &parallelismSizeThreshold);
                 auto terminalStateIndices =
                     init["terminal_state_indices"].cast<std::map<std::string, std::vector<int64_t>>>();
                 auto transitionInformation =
                     init["transition_information"].cast<std::map<std::string, std::vector<torch::Tensor>>>();
                 cMemoryData.initialize_data(transitionInformation, terminalStateIndices);
                 return cMemoryData;
               }),
           "Pickle method for C_MemoryData.",
           pybind11::return_value_policy::reference)
      .def("transitions", [](C_Memory::C_MemoryData &cMemoryData) {
             return cMemoryData.deref_transition_information_map();
           },
           pybind11::return_value_policy::reference)
      .def("terminal_state_indices",
           [](C_Memory::C_MemoryData &cMemoryData) {
             return cMemoryData.deref_terminal_state_indices();
           },
           pybind11::return_value_policy::reference);

  /*
   * This is the binding of STL std::map<std::string, torch::Tensor> aliased C_Memory_TensorMap.
   * C_Memory_TensorMap will be returned from the following functions:
   * - sample (in C_Memory)
   * - get_item (in C_Memory)
   */
  pybind11::bind_map<std::map<std::string, torch::Tensor>>(m, "C_Memory_TensorMap")
      .def("__repr__", [](std::map<std::string, torch::Tensor> &cMemoryTensorMap) {
        std::string reprString;
        std::stringstream ss;
        ss << &cMemoryTensorMap;
        reprString = "<C_Memory_TensorMap object at " + ss.str() + ">";
        return reprString;
      })
      .def(pybind11::pickle(
               [](std::map<std::string, torch::Tensor> &cMemoryTensorMap) {
                 pybind11::dict cMemoryTensorDict;
                 for (auto &pair : cMemoryTensorMap) {
                   cMemoryTensorDict[pair.first.c_str()] = pair.second;
                 }
                 return cMemoryTensorDict;
               },
               [](pybind11::dict &init) {
                 std::map<std::string, torch::Tensor> cMemoryTensorMap;
                 for (auto &pair : init) {
                   cMemoryTensorMap[pair.first.cast<std::string>()] = pair.second.cast<torch::Tensor>();
                 }
                 return cMemoryTensorMap;
               }),
           "Pickle method for C_Memory_TensorMap.",
           pybind11::return_value_policy::reference);
  /*
   * This is the binding of STL std::map<std::string, std::vector<torch::Tensor>>
   * aliased C_Memory_MapOfVectorOfTensors.
   * C_Memory_MapOfVectorOfTensors will be returned from the following functions:
   * - deref_transition_information_map (in C_Memory::C_MemoryData, access available as property
   * `transitions` in Python).
   */
  pybind11::bind_map<std::map<std::string, std::vector<torch::Tensor>>>(m,
                                                                        "C_Memory_MapOfVectorOfTensors")
      .def("__repr__", [](std::map<std::string, std::vector<torch::Tensor>> &cMemoryMapOfVectorOfTensors) {
        std::string reprString;
        std::stringstream ss;
        ss << &cMemoryMapOfVectorOfTensors;
        reprString = "<C_Memory_MapOfVectorOfTensors object at " + ss.str() + ">";
        return reprString;
      })
      .def(pybind11::pickle(
               [](std::map<std::string, std::vector<torch::Tensor>> &cMemoryMapOfVectorOfTensors) {
                 pybind11::dict cMemoryTensorDict;
                 for (auto &pair : cMemoryMapOfVectorOfTensors) {
                   cMemoryTensorDict[pair.first.c_str()] = pair.second;
                 }
                 return cMemoryTensorDict;
               },
               [](pybind11::dict &init) {
                 std::map<std::string, std::vector<torch::Tensor>> cMemoryMapOfVectorOfTensors;
                 for (auto &pair : init) {
                   cMemoryMapOfVectorOfTensors[pair.first.cast<std::string>()] =
                       pair.second.cast<std::vector<torch::Tensor>>();
                 }
                 return cMemoryMapOfVectorOfTensors;
               }),
           "Pickle method for C_Memory_MapOfVectorOfTensors.",
           pybind11::return_value_policy::reference);
  /*
   * This is the binding of STL std::map<std::string, std::vector<int64_t>> aliased C_Memory_MapOfVectorOfInt64.
   * C_Memory_MapOfVectorOfInt64 will be returned from the following functions:
   * - deref_terminal_state_indices (in C_Memory::C_MemoryData, access available as property
   *   `terminal_state_indices` in Python).
   */
  pybind11::bind_map<std::map<std::string, std::vector<int64_t>>>(m, "C_Memory_MapOfVectorOfInt64")
      .def("__repr__", [](std::map<std::string, std::vector<int64_t>> &cMemoryMapOfVectorOfInt64) {
        std::string reprString;
        std::stringstream ss;
        ss << &cMemoryMapOfVectorOfInt64;
        reprString = "<C_Memory_MapOfVectorOfTensors object at " + ss.str() + ">";
        return reprString;
      })
      .def(pybind11::pickle(
               [](std::map<std::string, std::vector<int64_t>> &cMemoryMapOfVectorOfInt64) {
                 pybind11::dict cMemoryTensorDict;
                 for (auto &pair : cMemoryMapOfVectorOfInt64) {
                   cMemoryTensorDict[pair.first.c_str()] = pair.second;
                 }
                 return cMemoryTensorDict;
               },
               [](pybind11::dict &init) {
                 std::map<std::string, std::vector<int64_t>> cMemoryMapOfVectorOfInt64;
                 for (auto &pair : init) {
                   cMemoryMapOfVectorOfInt64[
                       pair.first.cast<std::string>()] = pair.second.cast<std::vector<int64_t>>();
                 }
                 return cMemoryMapOfVectorOfInt64;
               }),
           "Pickle method for C_Memory_MapOfVectorOfInt64.",
           pybind11::return_value_policy::reference);
}
