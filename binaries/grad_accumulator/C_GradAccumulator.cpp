#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-loop-convert"
//
// Created by Kartik Rajeshwaran on 2022-12-14.
//

#include "C_GradAccumulator.h"

C_GradAccumulator::C_GradAccumulator(std::vector<std::string> &parameterKeys, int64_t boostStrapRounds) {
    /*!
     * Class constructor for C_GradAccumulator. This class reserves the memory for namedParametersGrads_ for gradient
     * accumulation. This is C++ backend equivalent to rlpack._C.grad_accumulator.GradAccumulator.__init__.
     * @param parameterKeys : parameter keys for the model for which gradient accumulation is being done.
     * @param boostStrapRounds :  The number of boostrap rounds over which accumulation and reduction is to take place.
     */
    parameterKeys_ = parameterKeys;
    bootstrapRounds_ = boostStrapRounds;
    namedParametersGrads_.reserve(bootstrapRounds_);
}

void C_GradAccumulator::accumulate(std::map<std::string, torch::Tensor> &namedParameters) {
    /*!
     * This method accumulates the gradient from the given named parameters. This method will throw error if
     * you attempt to accumulate more gradients than bootstrapRounds passed in class constructor.
     * This is C++ backend equivalent to rlpack._C.grad_accumulator.GradAccumulator.accumulate.
     * @param namedParameters : Map of named parameters.
     */
    if (namedParametersGrads_.size() == bootstrapRounds_) {
        throw std::overflow_error("Attempted to accumulate more than `bootstrap_rounds` (in Py; bootstrapRounds in C++)!");
    }
    std::map<std::string, torch::Tensor> namedParameterGrads;
    for (int64_t keyIndex = 0; keyIndex != parameterKeys_.size(); keyIndex++) {
        auto key = parameterKeys_[keyIndex];
        auto clonedGrad = namedParameters[key].grad().detach().clone();
        if (clonedGrad.isnan().all().item<bool>()) {
            throw std::runtime_error("Gradients were NaN! Did you call .backward() on loss?");
        }
        namedParameterGrads[key] = clonedGrad;
    }
    namedParametersGrads_.push_back(namedParameterGrads);
}

std::map<std::string, torch::Tensor> C_GradAccumulator::mean_reduce() {
    /*!
     * Performs mean reduction of accumulated gradients. This is C++ backend equivalent to
     * rlpack._C.grad_accumulator.GradAccumulator.mean_reduce.
     */
    if (namedParametersGrads_.empty()) {
        throw std::length_error("No gradients have been accumulated! Kindly accumulate gradients with `accumulate` method");
    }
    {
        // No grad scope
        torch::NoGradGuard no_grad_guard;
        for (int64_t namedParametersIndex = 0; namedParametersIndex != bootstrapRounds_; namedParametersIndex++) {
            for (int64_t keyIndex = 0; keyIndex != parameterKeys_.size(); keyIndex++) {
                auto key = parameterKeys_[keyIndex];
                auto param = namedParametersGrads_[namedParametersIndex][key];
                if (namedParametersIndex != 0) {
                    reducedParams_[key] += param / bootstrapRounds_;
                } else {
                    reducedParams_[key] = param / bootstrapRounds_;
                }
            }
        }
    }
    return reducedParams_;
}

std::map<std::string, torch::Tensor> C_GradAccumulator::sum_reduce() {
    /*!
     * Performs sum reduction of accumulated gradients. This is C++ backend equivalent to
     * rlpack._C.grad_accumulator.GradAccumulator.mean_reduce.
     */
    if (namedParametersGrads_.empty()) {
        throw std::length_error("No gradients have been accumulated! Kindly accumulate gradients with `accumulate` method");
    }
    {
        // No grad scope
        torch::NoGradGuard no_grad_guard;
        for (int64_t namedParametersIndex = 0; namedParametersIndex != bootstrapRounds_; namedParametersIndex++) {
            for (int64_t keyIndex = 0; keyIndex != parameterKeys_.size(); keyIndex++) {
                auto key = parameterKeys_[keyIndex];
                auto param = namedParametersGrads_[namedParametersIndex][key];
                if (namedParametersIndex != 0) {
                    reducedParams_[key] += param;
                } else {
                    reducedParams_[key] = param;
                }
            }
        }
    }
    return reducedParams_;
}

void C_GradAccumulator::clear() {
    /*!
     * Clears all the accumulated gradients. This is C++ backend equivalent to
     * rlpack._C.grad_accumulator.GradAccumulator.clear.
     */
    namedParametersGrads_.clear();
}

/*!
 * Default constructor C_GradAccumulator
 */
C_GradAccumulator::~C_GradAccumulator() = default;

#pragma clang diagnostic pop