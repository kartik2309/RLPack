//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "Adam.h"

namespace optimizer {
    Adam::Adam(const std::vector<torch::Tensor> &parameters,
               const std::shared_ptr<torch::optim::AdamOptions> &adamOptions
    ) : OptimizerBase(parameters, adamOptions) {
        std::shared_ptr<torch::optim::AdamOptions> adamOptions_ = std::make_shared<torch::optim::AdamOptions>(
                std::move(*adamOptions));
        optim = std::make_shared<torch::optim::Adam>(parameters, *adamOptions_);
    }

    torch::Tensor Adam::step(torch::optim::Optimizer::LossClosure closure) {
        return optim->step(closure);
    }

    float Adam::get_lr(int paramGroupIndex) {
        return (float) dynamic_cast<torch::optim::AdamOptions &>(optim->param_groups()[paramGroupIndex].options()).lr();
    }

    void Adam::set_lr(std::vector<float> &newLrs) const {
        for (int idx = 0; idx != newLrs.size(); idx++) {
            auto &options = dynamic_cast<torch::optim::AdamOptions &>(
                    optim->param_groups().at(idx).options()
            );
            float newLr = newLrs.at(idx);
            options.set_lr(newLr);
        }
    }

    uint32_t Adam::get_param_group_size() {
        return optim->param_groups().size();
    }
}