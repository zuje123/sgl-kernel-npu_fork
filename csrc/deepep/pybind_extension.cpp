#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <c10/util/ArrayRef.h>

#include "deep_ep.hpp"
#include "config.hpp"
#include "event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace py = pybind11;

at::TensorList pylist_to_tensorlist(const py::list& py_list) {
    std::vector<at::Tensor> vec = py::cast<std::vector<at::Tensor>>(py_list);
    return at::TensorList(vec);  // 构造 TensorList
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(), py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6, py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, std::string>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("get_notify_send_data", &deep_ep::Buffer::get_notify_send_data)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("notify_verify", &deep_ep::Buffer::notify_verify)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("fused_deep_moe", &deep_ep::Buffer::fused_deep_moe)
        .def("dispatch_ffn_combine",
            [](const deep_ep::Buffer& self, const at::Tensor &x, const at::Tensor &expertIds, const py::list &weight1_list,
               const py::list &scale1_list, const py::list &weight2_list, const py::list &scale2_list,
               const at::Tensor &expertScales, int64_t max_output_size, int64_t num_experts,
               int quant_mode) -> py::list {
                auto weight1 = py::cast<std::vector<at::Tensor>>(weight1_list);
                auto scale1 = py::cast<std::vector<at::Tensor>>(scale1_list);
                auto weight2 = py::cast<std::vector<at::Tensor>>(weight2_list);
                auto scale2 = py::cast<std::vector<at::Tensor>>(scale2_list);

                std::vector<at::Tensor> result = self.dispatch_ffn_combine(
                    x, expertIds, weight1, scale1, weight2, scale2, expertScales,
                    max_output_size, num_experts, quant_mode);

                py::list py_result;
                for (const auto& tensor : result) {
                    py_result.append(tensor);
                }
                return py_result;
            },
            py::arg("x"), py::arg("expert_ids"), py::arg("weight1") = py::list(), py::arg("scale1") = py::list(),
            py::arg("weight2") = py::list(), py::arg("scale2") = py::list(), py::arg("expert_scales"),
            py::arg("max_output_size"), py::arg("num_experts"), py::arg("quant_mode") = 0);
}
