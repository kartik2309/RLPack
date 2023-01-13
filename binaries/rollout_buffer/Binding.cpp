//
// Created by Kartik Rajeshwaran on 2022-12-26.
//


#include "../stl_bindings/StlBindings.h"
#include "C_RolloutBuffer.h"

PYBIND11_MODULE(C_RolloutBuffer, m) {
    /*!
     * Python bindings for C_RolloutBuffer. Only relevant public methods are exposed to Python.
     */
    m.doc() = "Module to provide Python binding for RolloutBuffer class";
    pybind11::class_<C_RolloutBuffer>(m, "C_RolloutBuffer")
            .def(pybind11::init<int64_t,
                                std::string &,
                                std::string &,
                                std::map<std::string, c10::intrusive_ptr<c10d::ProcessGroup>> &,
                                const std::chrono::duration<int32_t> &>(),
                 "Class constructor for C_RolloutBuffer.",
                 pybind11::arg("buffer_size"),
                 pybind11::arg("device"),
                 pybind11::arg("dtype"),
                 pybind11::arg("process_group_map"),
                 pybind11::arg("work_timeout"))
            .def("insert_transition",
                 &C_RolloutBuffer::insert_transition,
                 pybind11::arg("input_map"))
            .def("insert_policy_output",
                 &C_RolloutBuffer::insert_policy_output,
                 pybind11::arg("input_map"))
            .def("compute_returns",
                 &C_RolloutBuffer::compute_returns,
                 pybind11::arg("gamma"),
                 pybind11::return_value_policy::reference)
            .def("compute_discounted_td_residuals",
                 &C_RolloutBuffer::compute_discounted_td_residuals,
                 pybind11::arg("gamma"),
                 pybind11::return_value_policy::reference)
            .def("compute_generalized_advantage_estimates",
                 &C_RolloutBuffer::compute_generalized_advantage_estimates,
                 pybind11::arg("gamma"),
                 pybind11::arg("gae_lambda"),
                 pybind11::return_value_policy::reference)
            .def("get_stacked_states_current",
                 &C_RolloutBuffer::get_stacked_states_current,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_states_next",
                 &C_RolloutBuffer::get_stacked_states_next,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_rewards",
                 &C_RolloutBuffer::get_stacked_rewards,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_dones",
                 &C_RolloutBuffer::get_stacked_dones,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_action_log_probabilities",
                 &C_RolloutBuffer::get_stacked_action_log_probabilities,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_state_current_values",
                 &C_RolloutBuffer::get_stacked_state_current_values,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_state_next_values",
                 &C_RolloutBuffer::get_stacked_state_next_values,
                 pybind11::return_value_policy::reference)
            .def("get_stacked_entropies",
                 &C_RolloutBuffer::get_stacked_entropies,
                 pybind11::return_value_policy::reference)
            .def("get_states_statistics",
                 &C_RolloutBuffer::get_states_statistics,
                 pybind11::return_value_policy::reference)
            .def("get_advantage_statistics",
                 &C_RolloutBuffer::get_advantage_statistics,
                 pybind11::arg("gamma"),
                 pybind11::arg("gae_lambda"),
                 pybind11::return_value_policy::reference)
            .def("get_action_log_probabilities_statistics",
                 &C_RolloutBuffer::get_action_log_probabilities_statistics,
                 pybind11::return_value_policy::reference)
            .def("get_state_values_statistics",
                 &C_RolloutBuffer::get_state_values_statistics,
                 pybind11::return_value_policy::reference)
            .def("get_entropy_statistics",
                 &C_RolloutBuffer::get_entropy_statistics,
                 pybind11::return_value_policy::reference)
            .def("transition_at",
                 &C_RolloutBuffer::transition_at,
                 pybind11::arg("index"),
                 pybind11::return_value_policy::reference)
            .def("policy_output_at",
                 &C_RolloutBuffer::policy_output_at,
                 pybind11::arg("index"),
                 pybind11::return_value_policy::reference)
            .def("clear_transitions",
                 &C_RolloutBuffer::clear_transitions)
            .def("clear_policy_outputs",
                 &C_RolloutBuffer::clear_policy_outputs)
            .def("size_transitions",
                 &C_RolloutBuffer::size_transitions,
                 pybind11::return_value_policy::reference)
            .def("size_policy_outputs",
                 &C_RolloutBuffer::size_policy_outputs,
                 pybind11::return_value_policy::reference)
            .def("extend_transitions",
                 &C_RolloutBuffer::extend_transitions)
            .def("get_transitions_iterator",
                 [](C_RolloutBuffer &rolloutBuffer, int64_t batchSize){
                        rolloutBuffer.set_transitions_iterator(batchSize);
                        return pybind11::make_iterator(rolloutBuffer.get_dataloader_reference()->begin(),
                                                       rolloutBuffer.get_dataloader_reference()->end());
                    },
                    pybind11::arg("batch_size"),
                    pybind11::keep_alive<0, 1>()
                 );

    // Bind relevant StlBinding objects
    pybind11::bind_map<std::map<std::string, torch::Tensor>>(m, "TensorMap");
    pybind11::bind_map<std::map<std::string, c10::intrusive_ptr<c10d::ProcessGroup>>>(m, "ProcessGroupMap");
}