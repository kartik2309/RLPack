//
// Created by Kartik Rajeshwaran on 2022-07-21.
//

#ifndef RLPACK_OPS_H
#define RLPACK_OPS_H

#include <torch/torch.h>

class Ops {
public:
    static torch::Tensor vector_to_tensor(std::vector<std::vector<float_t>> &vector);

    static torch::Tensor vector_to_tensor(std::vector<std::vector<double_t>> &vector);

    template<typename returnType>
    static std::vector<std::vector<returnType>> tensor_to_vector(torch::Tensor &tensor);
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Template Function Definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
template <typename returnType>
std::vector<std::vector<returnType>> Ops::tensor_to_vector(torch::Tensor &tensor) {
    auto flattenedTensor = tensor.flatten();

    std::vector<returnType> vector_(flattenedTensor.data_ptr<returnType>(),
                                    flattenedTensor.data_ptr<returnType>() + flattenedTensor.numel());
    std::vector<returnType> shape_(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<std::vector<returnType>> vector = {vector_, shape_};
    return vector;
}



#endif //RLPACK_OPS_H
