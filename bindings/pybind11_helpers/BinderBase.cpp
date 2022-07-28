//
// Created by Kartik Rajeshwaran on 2022-07-11.
//

#include "BinderBase.h"

torch::Tensor BinderBase::pybind_array_to_torch_tensor(
        pybind11::array_t<float_t> &array, pybind11::tuple &shape,
        torch::ScalarType dataType, torch::DeviceType device
) {
    std::vector<int64_t> shapeVector;
    for (auto &shape_: shape) {
        shapeVector.push_back(shape_.cast<int64_t>());
    }

    auto tensor = torch::zeros(shapeVector);
    memmove(tensor.data_ptr(), array.data(), tensor.nbytes());

    auto tensorOptions = torch::TensorOptions().dtype(dataType).device(device);
    tensor = tensor.to(tensorOptions);
    return tensor;
}

torch::DeviceType BinderBase::get_device(std::string &device) {
    int deviceCode = deviceCodes_["device"];
    return deviceTypes_[deviceCode];
}

torch::ScalarType BinderBase::get_data_type(std::string &device) {
    int deviceCode = deviceCodes_["device"];
    return deviceDataTypes_[deviceCode];
}
