//
// Created by Kartik Rajeshwaran on 2022-07-05.
//

#ifndef RLPACK_NORMALIZATION_H
#define RLPACK_NORMALIZATION_H

#include <torch/torch.h>

class Normalization {
public:
    explicit Normalization(int32_t applyNorm = -1);

    ~Normalization();

    torch::Tensor apply_normalization(torch::Tensor &tensor, float_t eps = 5e-8, int32_t p = 2, int32_t dim = 0);

private:
    int32_t applyNorm_;

    static torch::Tensor min_max_normalization(torch::Tensor &tensor, float_t eps = 5e-8, int32_t dim = 0);

    static torch::Tensor standardization(torch::Tensor &tensor, float_t eps = 5e-8, int32_t dim = 0);

    static torch::Tensor p_normalization(torch::Tensor &tensor, float_t eps = 5e-8, int32_t p = 2, int32_t dim = 0);
};


#endif //RLPACK_NORMALIZATION_H
