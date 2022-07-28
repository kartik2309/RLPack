//
// Created by Kartik Rajeshwaran on 2022-07-12.
//

#include "ModelBase.h"

namespace model {

    torch::Tensor ModelBase::forward(torch::Tensor x) {
        return x;
    }

    ModelBase::ModelBase() = default;


    ModelBase::~ModelBase() = default;
}