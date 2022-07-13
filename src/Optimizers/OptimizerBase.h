//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#ifndef RLPACK_OPTIMIZERBASE_H
#define RLPACK_OPTIMIZERBASE_H

#include <torch/optim.h>
#include <boost/log/trivial.hpp>

namespace optimizer {
    class OptimizerBase : public torch::optim::Optimizer {
    public:
        std::shared_ptr<torch::optim::Optimizer> optim;

        explicit OptimizerBase(
                const std::vector<torch::Tensor> &parameters,
                const std::shared_ptr<torch::optim::OptimizerOptions> &defaults
        );

        ~OptimizerBase() override;

        torch::Tensor step(torch::optim::Optimizer::LossClosure closure) override = 0;
    };
}


#endif //RLPACK_OPTIMIZERBASE_H
