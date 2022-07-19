//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_MODELBASE_H
#define RLPACK_MODELBASE_H

#include <torch/torch.h>
#include "../../Options/ModelOptions/ModelOptionsBase.h"

namespace model {
    class ModelBase : public torch::nn::Module {

    public:

        ModelBase();

        virtual torch::Tensor forward(torch::Tensor x);

        ~ModelBase() override;
    };

}// namespace model


#endif //RLPACK_MODELBASE_H
