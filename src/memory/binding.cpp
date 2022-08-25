//
// Created by Kartik Rajeshwaran on 2022-08-22.
//

#include "C_Memory.h"

PYBIND11_MAKE_OPAQUE(std::vector<torch::Tensor>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::vector<torch::Tensor> *>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::vector<torch::Tensor>>)
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>)
PYBIND11_MODULE(C_Memory, m) {
    m.doc() = "Module to provide Python binding for C_Memory class";
    pybind11::class_<C_Memory>(m, "C_Memory")
            .def(pybind11::init<pybind11::int_ &, pybind11::str &>())
            .def("insert", &C_Memory::insert, "Insertion method to memory.",
                 pybind11::arg("state_current"),
                 pybind11::arg("state_next"),
                 pybind11::arg("reward"),
                 pybind11::arg("action"),
                 pybind11::arg("done"),
                 pybind11::arg("is_terminal"))
            .def("reserve", &C_Memory::reserve, "Reserve method to reserve the memory size.",
                 pybind11::arg("buffer_size"))
            .def("get_item", &C_Memory::get_item, "Get item method to get an item as per index.",
                 pybind11::return_value_policy::reference,
                 pybind11::arg("index"))
            .def("delete_item", &C_Memory::delete_item, "Delete item method to delete an item as per index.",
                 pybind11::arg("index"))
            .def("sample", pybind11::overload_cast<int32_t, float_t>(&C_Memory::sample),
                 "Sample items from memory."
                 " Overload for when we pass both batchSize and forceTerminalStateProbability."
                 " This method samples items and arranges them quantity-wise.",
                 pybind11::return_value_policy::reference)
            .def("sample", pybind11::overload_cast<int32_t>(&C_Memory::sample),
                 "Sample items from memory."
                 " Overload for when we pass only batchSize."
                 " This method samples items and arranges them quantity-wise.",
                 pybind11::return_value_policy::reference)
            .def("initialize", &C_Memory::initialize, "Initialize the Memory with input vector of values.")
            .def("clear", &C_Memory::clear, "Clear all items in memory.")
            .def("size", &C_Memory::size, "Return the size of the memory.",
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
                 pybind11::return_value_policy::copy)
            .def("__len__", [](C_Memory &cMemory) { return cMemory.size(); },
                 pybind11::return_value_policy::reference)
            .def(pybind11::pickle(
                         // __getstate__ method
                         [](C_Memory &cMemory) { return cMemory.view(); },
                         // __setstate__ method
                         [](C_Memory::C_MemoryData &init) {
                             C_Memory cMemory;
                             cMemory.initialize(init);
                             return cMemory;
                         }),
                 "Pickle method for C_Memory.",
                 pybind11::return_value_policy::reference
            )
            .def_property("c_memory_data", [](C_Memory &cMemory) {
                              return cMemory.cMemoryData;
                          },
                          nullptr, pybind11::return_value_policy::reference);

    pybind11::class_<C_Memory::C_MemoryData>(m, "C_MemoryData")
            .def("__repr__", [](C_Memory::C_MemoryData &cMemoryData) {
                std::string reprString;
                std::stringstream ss;
                ss << &cMemoryData;
                reprString = "<C_Memory::C_MemoryData object at " + ss.str() + ">";
                return reprString;
            })
            .def(pybind11::pickle(
                         // __getstate__ method
                         [](C_Memory::C_MemoryData &cMemoryData) {
                             pybind11::dict cMemoryDataDict;
                             cMemoryDataDict["data"] = cMemoryData.derefCoreData();
                             cMemoryDataDict["terminal_state_indices"] = cMemoryData.derefTerminalStateIndices();
                             return cMemoryDataDict;
                         },
                         // __setstate__ method
                         [](pybind11::dict &init) {
                             C_Memory::C_MemoryData cMemoryData;
                             auto terminalStateIndices = init["terminal_state_indices"].cast<
                                     std::vector<int64_t>>();
                             cMemoryData.terminalStatesIndicesPtr = &terminalStateIndices;
                             auto coreData = init["data"].cast<std::map<std::string,
                                     std::vector<torch::Tensor>>>();
                             for (auto &pair: coreData) {
                                 cMemoryData.coreDataPtr[pair.first] = &pair.second;
                             }
                             return cMemoryData;
                         }),
                 "Pickle method for C_MemoryData.",
                 pybind11::return_value_policy::reference)
            .def_property("_data_ref", [](C_Memory::C_MemoryData &cMemoryData) {
                return cMemoryData.coreDataPtr;
            }, nullptr, pybind11::return_value_policy::reference)
            .def_property("data_deref", [](C_Memory::C_MemoryData &cMemoryData) {
                              return cMemoryData.derefCoreData();
                          }, nullptr,
                          pybind11::return_value_policy::reference)
            .def_property("terminal_state_indices_deref",
                          [](C_Memory::C_MemoryData &cMemoryData) {
                              return cMemoryData.derefTerminalStateIndices();
                          }, nullptr,
                          pybind11::return_value_policy::reference);

    pybind11::bind_map<std::map<std::string, torch::Tensor>>(m, "MapOfTensors")
            .def("__repr__", [](std::map<std::string, torch::Tensor> &MapOfTensors) {
                std::string reprString;
                std::stringstream ss;
                ss << &MapOfTensors;
                reprString = "<MapOfTensors object at " + ss.str() + ">";
                return reprString;
            });
    pybind11::bind_vector<std::vector<int64_t>>(m, "VectorOfInt64")
            .def("__repr__", [](std::vector<int64_t> &vectorOfInt64) {
                std::string reprString;
                std::stringstream ss;
                ss << &vectorOfInt64;
                reprString = "<VectorOfInt64 object at " + ss.str() + ">";
                return reprString;
            });
    pybind11::bind_map<std::map<std::string, std::vector<torch::Tensor>>>(m, "C_MemoryDataMap")
            .def("__repr__", [](std::map<std::string, std::vector<torch::Tensor>> &c_MemoryDataMap) {
                std::string reprString;
                std::stringstream ss;
                ss << &c_MemoryDataMap;
                reprString = "<C_MemoryDataMap object at " + ss.str() + ">";
                return reprString;
            });
    pybind11::bind_map<std::map<std::string, std::vector<torch::Tensor> *>>(m, "C_MemoryDataMapPtr")
            .def("__repr__", [](std::map<std::string,
                    std::vector<torch::Tensor> *> &c_MemoryDataMapPtr) {
                std::string reprString;
                std::stringstream ss;
                ss << &c_MemoryDataMapPtr;
                reprString = "<C_MemoryDataMapPtr object at " + ss.str() + ">";
                return reprString;
            });
}