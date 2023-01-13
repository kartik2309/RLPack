
#include "../stl_bindings/StlBindings.h"
#include "C_ReplayBuffer.cuh"


/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup replay_buffer_group replay_buffer
 * @brief The C++ backend for rlpack._C.replay_buffer.ReplayBuffer class. Heavier workloads have been optimized
 * with multithreading with OpenMP and CUDA (if CUDA compatible device is found).
 * @{
 */
PYBIND11_MODULE(C_ReplayBuffer, m) {
    /*!
     * Python bindings for C_ReplayBuffer, C_ReplayBufferData and all the opaque objects. All bindings are pickleable.
     * Python binding for C_ReplayBuffer class. Only relevant methods are exposed to Python
     */
    m.doc() = "Module to provide Python binding for C_ReplayBuffer class";
    pybind11::class_<C_ReplayBuffer>(m, "C_ReplayBuffer")
            .def(pybind11::init<int64_t, std::string &, std::string &, int32_t, int32_t>(),
                 "Class constructor for C_ReplayBuffer",
                 pybind11::arg("buffer_size"),
                 pybind11::arg("device"),
                 pybind11::arg("dtype"),
                 pybind11::arg("prioritization_strategy_code"),
                 pybind11::arg("batch_size"))
            .def("insert_policy_output", &C_ReplayBuffer::insert,
                 "Insertion method to memory.",
                 pybind11::arg("state_current"),
                 pybind11::arg("state_next"),
                 pybind11::arg("reward"),
                 pybind11::arg("action"),
                 pybind11::arg("done"),
                 pybind11::arg("priority"),
                 pybind11::arg("probability"),
                 pybind11::arg("weight"),
                 pybind11::arg("is_terminal"))
            .def("get_item", &C_ReplayBuffer::get_item,
                 "Get item method to get an item as per index.",
                 pybind11::return_value_policy::reference,
                 pybind11::arg("index"))
            .def("set_item", &C_ReplayBuffer::set_item,
                 "Set Item method to set item at an index.",
                 pybind11::arg("index"),
                 pybind11::arg("state_current"),
                 pybind11::arg("state_next"),
                 pybind11::arg("reward"),
                 pybind11::arg("action"),
                 pybind11::arg("done"),
                 pybind11::arg("priority"),
                 pybind11::arg("probability"),
                 pybind11::arg("weight"),
                 pybind11::arg("is_terminal"))
            .def("delete_item", &C_ReplayBuffer::delete_item,
                 "Delete item method to delete an item as per index.",
                 pybind11::arg("index"))
            .def("sample", &C_ReplayBuffer::sample,
                 "Sample items from memory. This method samples items and arranges them quantity-wise.",
                 pybind11::arg("force_terminal_state_probability"),
                 pybind11::arg("parallelism_size_threshold"),
                 pybind11::arg("alpha"),
                 pybind11::arg("beta"),
                 pybind11::arg("num_segments"),
                 pybind11::return_value_policy::reference)
            .def("update_priorities", &C_ReplayBuffer::update_priorities,
                 "Method to update priorities and associated prioritization values",
                 pybind11::arg("random_indices"),
                 pybind11::arg("new_priorities"))
            .def("initialize", &C_ReplayBuffer::initialize,
                 "Initialize the Memory with input vector of values.",
                 pybind11::arg("c_memory_data"))
            .def("clear", &C_ReplayBuffer::clear,
                 "Clear all items in memory.")
            .def("size", &C_ReplayBuffer::size,
                 "Return the size of the memory.",
                 pybind11::return_value_policy::reference)
            .def("num_terminal_states", &C_ReplayBuffer::num_terminal_states,
                 "Returns the number of terminal accumulated so far",
                 pybind11::return_value_policy::reference)
            .def("tree_height", &C_ReplayBuffer::tree_height,
                 "Returns the height of the tree. Relevant only if using prioritized memory.",
                 pybind11::return_value_policy::reference)
            .def("view", &C_ReplayBuffer::view,
                 "Return the current memory view.",
                 pybind11::return_value_policy::reference)
            .def(
                    "__repr__", [](C_ReplayBuffer &cReplayBuffer) {
                        std::string reprString;
                        std::stringstream ss;
                        ss << &cReplayBuffer;
                        reprString = "<C_ReplayBuffer object at " + ss.str() + ">";
                        return reprString;
                    },
                    pybind11::return_value_policy::reference)
            .def(pybind11::pickle([](C_ReplayBuffer &cReplayBuffer) { return cReplayBuffer.view(); }, [](C_ReplayBufferData &init) {
                 auto cReplayBuffer = new C_ReplayBuffer();
                 cReplayBuffer->initialize(init);
                 return *cReplayBuffer; }), "Pickle method for C_ReplayBuffer.", pybind11::return_value_policy::reference);
    /*
     * Python binding for C_ReplayBufferData class. Only relevant methods are exposed to Python.
     * This will be available only via C_ReplayBuffer object.
     */
    pybind11::class_<C_ReplayBufferData>(m, "C_ReplayBufferData")
            .def("__repr__", [](C_ReplayBufferData &cReplayBufferData) {
                std::string reprString;
                std::stringstream ss;
                ss << &cReplayBufferData;
                reprString = "<C_ReplayBuffer::C_ReplayBufferData object at " + ss.str() + ">";
                return reprString;
            })
            .def(
                    "transition_information", [](C_ReplayBufferData &cReplayBufferData) {
                        return cReplayBufferData.dereference_transition_information();
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "terminal_state_indices", [](C_ReplayBufferData &cReplayBufferData) {
                        return cReplayBufferData.dereference_terminal_state_indices();
                    },
                    pybind11::return_value_policy::reference)
            .def(
                    "priorities", [](C_ReplayBufferData &cReplayBufferData) {
                        return cReplayBufferData.dereference_priorities();
                    },
                    pybind11::return_value_policy::reference)
            .def(pybind11::pickle([](C_ReplayBufferData &cReplayBufferData) {
                 pybind11::dict cMemoryDataDict;
                 cMemoryDataDict["transition_information"] = cReplayBufferData.dereference_transition_information();
                 cMemoryDataDict["terminal_state_indices"] = cReplayBufferData.dereference_terminal_state_indices();
                 cMemoryDataDict["priorities"] = cReplayBufferData.dereference_priorities();
                 return cMemoryDataDict; }, [](pybind11::dict &init) {
                 auto *cReplayBufferData = new C_ReplayBufferData();
                 auto transitionInformation = init["transition_information"]
                     .cast<std::map<std::string, std::deque<torch::Tensor>>>();
                 for (auto &pair : transitionInformation) {
                   auto key = pair.first;
                   auto data = pair.second;
                   auto dataDynamicallyAllocated = new std::deque<torch::Tensor>(data.begin(), data.end());
                   cReplayBufferData->set_transition_information_references(key, dataDynamicallyAllocated);
                 }
                 auto terminalStateIndices = init["terminal_state_indices"]
                     .cast<std::map<std::string, std::deque<int64_t>>>()["terminal_state_indices"];
                 auto *terminalStateIndicesDynamicallyAllocated = new std::deque<int64_t>(
                     terminalStateIndices.begin(),
                     terminalStateIndices.end());
                 cReplayBufferData->set_terminal_state_indices_reference(terminalStateIndicesDynamicallyAllocated);
                 auto priorities = init["priorities"]
                     .cast<std::map<std::string, std::deque<float_t>>>()["priorities"];
                 auto *prioritiesDynamicallyAllocated = new std::deque<float_t>(priorities.begin(),
                                                                                priorities.end());
                 std::copy(priorities.begin(), priorities.end(),
                           prioritiesDynamicallyAllocated->begin());
                 cReplayBufferData->set_priorities_reference(prioritiesDynamicallyAllocated);
                 return *cReplayBufferData; }), "Pickle method for C_ReplayBufferData.", pybind11::return_value_policy::reference);

    // Bind relevant StlBinding objects
    pybind11::bind_map<std::map<std::string, torch::Tensor>>(m, "TensorMap");
    pybind11::bind_map<std::map<std::string, std::deque<torch::Tensor>>>(m, "TensorDequeMap");
    pybind11::bind_map<std::map<std::string, std::deque<int64_t>>>(m, "Int64DequeMap");
}
/*!
 * @} @I{ // End group replay_buffer_group }
 * @} @I{ // End group binaries_group }
 */
