//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Sgd.h"

namespace optimizer {
    Sgd::Sgd(const std::vector<torch::Tensor> &parameters,
             const std::shared_ptr<torch::optim::SGDOptions> &sgdOptions
    ) : OptimizerBase(parameters, sgdOptions) {
        optim = std::make_shared<torch::optim::SGD>(parameters, *sgdOptions);

    }

    torch::Tensor Sgd::step(torch::optim::Optimizer::LossClosure closure) {
        return optim->step(closure);
    }

    float Sgd::get_lr(int paramGroupIndex) {
        return (float) dynamic_cast<torch::optim::SGDOptions &>(optim->param_groups()[0].options()).lr();
    }

    void Sgd::set_lr(std::vector<float> &newLrVector) const {
        for (int idx = 0; idx != newLrVector.size(); idx++) {
            auto &options = dynamic_cast<torch::optim::SGDOptions &>(
                    optim->param_groups().at(idx).options()
            );
            float newLr = newLrVector.at(idx);
            options.set_lr(newLr);
        }
    }

    uint32_t Sgd::get_param_group_size() {
        return optim->param_groups().size();
    }
}
