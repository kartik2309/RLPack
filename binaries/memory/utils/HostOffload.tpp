
#ifndef RLPACK_BINARIES_MEMORY_UTILS_HOSTOFFLOAD_TPP_
#define RLPACK_BINARIES_MEMORY_UTILS_HOSTOFFLOAD_TPP_

#ifndef __CUDA_AVAILABLE__

#include <omp.h>

#include <cmath>
#include <random>

#include "../../utils/ops/arg_mergesort.cuh"

/*!
 * @addtogroup binaries_group binaries
 * @brief Binaries Module consists of C++ backend exposed via pybind11 to rlpack via rlpack._C. These modules are
 * optimized to perform heavier workloads.
 * @{
 * @addtogroup memory_group memory
 * @brief Memory module is the C++ backend for rlpack._C.memory.Memory class. Heavier workloads have been optimized
 * with multithreading with OpenMP and CUDA (if CUDA compatible device is found).
 * @{
 * @addtogroup offload_group offload
 * @brief Template class to offload some heavier computation to specialised hardware. Functions will be executed in CUDA
 * if CUDA device is available else OpenMP routines will be used to execute the functions on CPU.
 * @{
 * @defgroup cpu_group cpu
 * @brief CPU optimized implementation for Offload
 * @{
 */
/*!
 * @class Offload
 * @brief Template Offload class for CPU with CPU optimized OpenMP routines
 */
template<typename DType>
class Offload {

public:
    //! The vector to store final results.
    std::vector<DType> result;

    explicit Offload(int64_t bufferSize);
    ~Offload();

    template<class Container>
    DType cumulative_sum(const Container &inputContainer, int64_t parallelismSizeThreshold);

    template<class Container>
    void shuffle(const Container &inputContainer, int64_t parallelismSizeThreshold);

    void generate_priority_seeds(DType cumulativeSum,
                                 int64_t parallelismSizeThreshold,
                                 uint64_t startPoint = 0.0);

    template<class Container>
    void arg_quantile_segment_indices(int64_t numSegments,
                                      const Container &inputContainer,
                                      int64_t parallelismSizeThreshold);

    void reset();

private:
    //! Float pointer to an array to store random errors.
    float_t *errorArray_;
    //! Int64 pointer to an array to store indices.
    uint64_t *indexArray_;
    //! DType pointer to an array to store input container's data to be processed.
    DType *inputContainerData_;
    //! The Vector to store unique priorities. This is used in Offload::arg_quantile_segment_indices method.
    std::vector<DType> uniquePriorities_;
    //! The Vector to store intermediate values to be shuffled.
    std::vector<DType> toShuffleVector_;
    //! The Vector to store frequencies of each priority. This is used in Offload::arg_quantile_segment_indices method.
    std::vector<int64_t> priorityFrequencies_;
};
/*!
 * @} @I{ // End group cpu_group }
 * @} @I{ // End group offload_group }
 * @} @I{ // End group memory_group }
 * @} @I{ // End group binaries_group }
 */


template<typename DType>
Offload<DType>::Offload(int64_t bufferSize) {
    /*!
     * Constructor for Offload. Dynamically allocates required memory and initialises necessary variables.
     */
    errorArray_ = new float_t[bufferSize];
    indexArray_ = new uint64_t[bufferSize];
    inputContainerData_ = new DType[bufferSize];
    uniquePriorities_.reserve(bufferSize);
    priorityFrequencies_.reserve(bufferSize);
    result = std::vector<DType>(bufferSize);
    toShuffleVector_ = std::vector<DType>(bufferSize);
}

template<typename DType>
Offload<DType>::~Offload() {
    /*!
     * Destructor for Offload. De-allocated all dynamically allocated memory.
     */
    delete[] errorArray_;
    delete[] indexArray_;
    delete[] inputContainerData_;
}

template<typename DType>
template<class Container>
DType Offload<DType>::cumulative_sum(const Container &inputContainer, int64_t parallelismSizeThreshold) {
    /*!
     * This template method computes the cumulative sum of a given input container.
     *
     *
     * Template Arguments
     * - Container: Must be an STL Container initialised with DType, hence must have implemented `size` method.
     *
     * @param inputContainer : The input container for which sum has to be computed.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP
     * parallelized routines are to be used.
     * @return The cumulative sum.
     */
    DType cumulativeSum = 0;
    size_t numElements = inputContainer.size();
    bool enableParallelism = parallelismSizeThreshold < inputContainer.size();
    {
        // Compute sum with reduction.
#pragma omp parallel for if (enableParallelism) default(none) \
        firstprivate(inputContainer, numElements)             \
                reduction(+                                   \
                          : cumulativeSum)                    \
                        schedule(static)
        for (uint64_t index = 0; index < numElements; index++) {
            auto priority = inputContainer[index];
            cumulativeSum += priority;
        }
    }
    return cumulativeSum;
}

template<typename DType>
template<class Container>
void Offload<DType>::shuffle(const Container &inputContainer, int64_t parallelismSizeThreshold) {
    /*!
     * This template method shuffles the given input container. Results are flushed into Offload::shuffle
     *
     *
     * Template Arguments
     * - Container: Must be an STL Container initialised with DType, hence must have implemented `size` method.
     *
     * @param inputContainer : The input container which has to be shuffled.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP parallelized
     * routines are to be used.
     */
    std::random_device rd;
    std::mt19937 generator(rd());
    int64_t numElements = inputContainer.size();
    std::uniform_real_distribution<float_t> randomErrorDistribution(0, static_cast<float_t>(numElements) - 1);
    bool enableParallelism = parallelismSizeThreshold < numElements;
    {
#pragma omp parallel for if (enableParallelism) default(none)         \
        firstprivate(randomErrorDistribution, generator, numElements) \
                shared(errorArray_, indexArray_)
        for (uint64_t index = 0; index < numElements; index++) {
            errorArray_[index] = randomErrorDistribution(generator);
            indexArray_[index] = index;
        }
    }
    arg_mergesort(errorArray_, indexArray_, 0, numElements - 1, parallelismSizeThreshold);
    {
#pragma omp parallel for if (enableParallelism) default(none)  \
        firstprivate(inputContainer, numElements, indexArray_) \
                shared(result)
        for (uint64_t index = 0; index < numElements; index++) {
            result[index] = inputContainer[indexArray_[index]];
        }
    }
}

template<typename DType>
void Offload<DType>::generate_priority_seeds(DType cumulativeSum, int64_t parallelismSizeThreshold, uint64_t startPoint) {
    /*!
     * This method generates the priority seeds. This is used by C_Memory::sample when using
     * proportional prioritization strategy. This method generates seeds between arguments
     * startPoint and cumulativeSum. Results are flushed into Offload::result.
     *
     * @param cumulativeSum : The cumulative sum upto which the seeds are to be generated.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP parallelized
     * routines are to be used.
     * @param startPoint : The start point, i.e. the smallest value of the generated seeds.
     */
    std::random_device rd;
    std::mt19937 generator(rd());
    auto numElements = static_cast<size_t>(cumulativeSum);
    std::uniform_real_distribution<float_t> randomErrorDistribution(0, static_cast<float_t>(numElements) - 1);
    bool enableParallelism = parallelismSizeThreshold < numElements;

    {
#pragma omp parallel for if (enableParallelism) default(none)                     \
        firstprivate(randomErrorDistribution, generator, numElements, startPoint) \
                shared(toShuffleVector_)
        for (uint64_t index = 0; index < numElements; index++) {
            toShuffleVector_[index] =
                    static_cast<float_t>(startPoint) + static_cast<float_t>(index) + randomErrorDistribution(generator);
        }
    }
    shuffle(toShuffleVector_, parallelismSizeThreshold);
}

template<typename DType>
template<class Container>
void Offload<DType>::arg_quantile_segment_indices(int64_t numSegments,
                                                  const Container &inputContainer,
                                                  int64_t parallelismSizeThreshold) {
    /*!
     * The template method generates the quantile segments for given input container and the resulting
     * indices of inputContainer from this operation is flushed into Offload::result.
     *
     *
     * Template Arguments
     * - Container: Must be an STL Container initialised with DType, hence must have implemented `size` method.
     *
     * @param numSegments : The number of segments to be divided inputContainer into.
     * @param inputContainer : The input container which is to be divided into `numSegments` segments.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP parallelized
     * routines are to be used.
     */
    size_t numElements = inputContainer.size();
    bool enableParallelism = parallelismSizeThreshold < numElements;
    {
#pragma omp parallel for if (enableParallelism) default(none) \
        firstprivate(numElements, inputContainer)             \
                shared(inputContainerData_, indexArray_)
        for (uint64_t index = 0; index < numElements; index++) {
            inputContainerData_[index] = inputContainer[index];
            indexArray_[index] = index;
        }
    }
    arg_mergesort(inputContainerData_, indexArray_, 0, numElements - 1, parallelismSizeThreshold);
    for (uint64_t index = 0; index < numElements; index++) {
        auto priority = inputContainerData_[index];
        if (uniquePriorities_.empty() or uniquePriorities_.back() != priority) {
            uniquePriorities_.push_back(priority);
            priorityFrequencies_.push_back(1);
            continue;
        }
        priorityFrequencies_.back() += 1;
    }
    // Cumulative sun of the frequencies.
    auto cumulativeFrequencySum = cumulative_sum(priorityFrequencies_, parallelismSizeThreshold);
    // `segmentsInfo` will store quantile index as a pair of sorted indices, original indices.
    for (int64_t index = 1; index < numSegments + 1; index++) {
        int64_t quantileIndex = ceil(
                cumulativeFrequencySum * (static_cast<float_t>(index) / static_cast<float_t>(numSegments)));
        result[index - 1] = indexArray_[quantileIndex - 1];
    }
    uniquePriorities_.clear();
    priorityFrequencies_.clear();
}

template<typename DType>
void Offload<DType>::reset() {
    /*!
     * Reset method for Offload. This will clear the Offload::result vector and fills it with 0.
     */
    {
#pragma omp parallel for default(none) shared(result)
        for (uint64_t index = 0; index < result.size(); index++) {
            result[index] = 0;
        }
    }
}

#endif//__CUDA_AVAILABLE__
#endif//RLPACK_BINARIES_MEMORY_UTILS_HOSTOFFLOAD_TPP_
