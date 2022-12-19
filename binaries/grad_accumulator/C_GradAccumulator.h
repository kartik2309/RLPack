//
// Created by Kartik Rajeshwaran on 2022-12-14.
//

#ifndef RLPACK_BINARIES_GRAD_ACCUMULATOR_CGRADACCUMULATOR_H_
#define RLPACK_BINARIES_GRAD_ACCUMULATOR_CGRADACCUMULATOR_H_

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <string>
#include <vector>

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup grad_accumulator_group grad_accumulator
 * @brief Memory module is the C++ backend for rlpack._C.grad_accumulator.GradAccumulator class.
 * @{
 */
class C_GradAccumulator {
public:
    C_GradAccumulator(std::vector<std::string> &parameterKeys, int64_t boostStrapRounds);
    ~C_GradAccumulator();

    void accumulate(std::map<std::string, torch::Tensor> &namedParameters);
    std::map<std::string, torch::Tensor> mean_reduce();
    std::map<std::string, torch::Tensor> sum_reduce();
    std::map<std::string, torch::Tensor> get_item(int64_t index);
    void set_item(int64_t index, std::map<std::string, torch::Tensor> &namedParameters);
    void delete_item(int64_t index);
    size_t size();
    void clear();

private:
    //! The number of boostrap rounds over which accumulation and reduction is to take place.
    int64_t bootstrapRounds_;
    //! The parameter keys for the model for which gradient accumulation is being done.
    std::vector<std::string> parameterKeys_;
    //! The vector to accumulate the gradients.
    std::vector<std::map<std::string, torch::Tensor>> namedParametersGrads_;
    //! The map to store final results of reduced parameters.
    std::map<std::string, torch::Tensor> reducedParams_;
};
/*!
 * @} @I{ // End group grad_accumulator_group }
 * @} @I{ // End group binaries_group }
 */


#endif//RLPACK_BINARIES_GRAD_ACCUMULATOR_CGRADACCUMULATOR_H_
