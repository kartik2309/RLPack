//
// Created by Kartik Rajeshwaran on 2022-11-15.
//

#ifndef RLPACK_BINARIES_MEMORY_UTILS_HOSTOFFLOAD_TPP_
#define RLPACK_BINARIES_MEMORY_UTILS_HOSTOFFLOAD_TPP_

#ifndef __CUDA_AVAILABLE__
#include <omp.h>

#include <cmath>
#include <limits>
#include <random>

#include "../../utils/ops/arg_mergesort.cuh"

template<typename DType>
class Offload {
public:
    explicit Offload(int64_t bufferSize, int32_t batchSize);
    ~Offload();

    std::vector<DType> result;

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
private:
    float_t *errorArray_;
    uint64_t *indexArray_;
    DType *inputContainerData_;
    std::vector<DType> uniquePriorities_;
    std::vector<int64_t> priorityFrequencies_;
    std::vector<DType> toShuffleVector_;
};

template<typename DType>
Offload<DType>::Offload(int64_t bufferSize, int32_t batchSize) {
    errorArray_ = new float_t[bufferSize];
    indexArray_ = new uint64_t[bufferSize];
    inputContainerData_ = new DType[bufferSize];
    uniquePriorities_.reserve(bufferSize);
    priorityFrequencies_.reserve(bufferSize);
    result = std::vector<DType>(bufferSize);
    toShuffleVector_ = std::vector<DType>(batchSize);
}

template<typename DType>
Offload<DType>::~Offload() {
    delete[] errorArray_;
    delete[] indexArray_;
    delete[] inputContainerData_;
}

template<typename DType>
template<class Container>
DType Offload<DType>::cumulative_sum(const Container &inputContainer, int64_t parallelismSizeThreshold) {
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

#endif//__CUDA_AVAILABLE__
#endif//RLPACK_BINARIES_MEMORY_UTILS_HOSTOFFLOAD_TPP_
