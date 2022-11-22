//
// Created by Kartik Rajeshwaran on 2022-11-15.
//

#ifndef RLPACK_BINARIES_MEMORY_UTILS_CUDAOFFLOAD_CUH_
#define RLPACK_BINARIES_MEMORY_UTILS_CUDAOFFLOAD_CUH_

#ifdef __CUDA_AVAILABLE__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <random>
#include <vector>

#include "../../utils/ops/arg_mergesort.cuh"

template<typename C>
struct is_vector : std::false_type {
};
template<typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {
};
template<typename C>
inline constexpr bool match_vector_type = is_vector<C>::value;

#define MAX_NUM_BLOCKS 256
#define MAX_REDUCTION_BLOCKS 2
#define NUM_THREADS 1024
#define MAX_SHUFFLE_ROUNDS 1024
#define MAX_SHARED_MEMORY_BLOCKS 8
#define BUFFERSIZE_FACTOR 7

__shared__ float_t G_sharedData[MAX_SHARED_MEMORY_BLOCKS][NUM_THREADS];

template<typename DType>
__device__ void cumulative_sum_device_fn(DType *sum, const DType *vector);

template<typename DType>
__global__ void cumulative_sum_kernel(DType *sums,
                                      const DType *data,
                                      size_t size,
                                      int64_t numBlocks);

template<typename DType>
__global__ void random_setup_kernel(uint64_t *sortedIndices,
                                    DType *sortedRandomValues,
                                    uint64_t seed,
                                    curandState *cuRandStates,
                                    uint64_t startPoint,
                                    size_t size);

template<typename DType>
__global__ void shuffle_kernel(DType *shuffle, const uint64_t *shuffledIndices, size_t size, uint64_t startPoint);

template<typename DType>
__global__ void array_index_fill_with_random_uniform_error_kernel(DType *fillArray,
                                                                  uint64_t start,
                                                                  uint64_t seed,
                                                                  curandState *cuRandStates,
                                                                  size_t size);

template<typename DType>
__global__ void array_index_fill_kernel(DType *fillArray,
                                        uint64_t start,
                                        size_t size);


template<typename DType>
class Offload {
public:
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
    float_t *errorArrayHost_, *errorArrayDevice_;
    uint64_t *indexArrayHost_, *indexArrayDevice_;
    DType *inputContainerDataHost_, *inputContainerDataDevice_,
            *sumsHost_, *sumsDevice_;
    curandState *statesHost_, *statesDevice_;
    std::vector<DType> uniquePriorities_;
    std::vector<int64_t> priorityFrequencies_;
    std::vector<DType> toShuffleVector_;
    uint64_t allocatedCapacity_;

    void shuffle_(size_t numElements, int64_t parallelismSizeThreshold);

    void execute_random_setup_kernel(size_t numElements, uint64_t startPoint, int64_t parallelismSizeThreshold);

    void execute_shuffle_kernel(size_t numElements, uint64_t startPoint, int64_t parallelismSizeThreshold);

    template<class Container>
    size_t fill_input_data(Container &inputContainer, int64_t parallelismSizeThreshold);

    void cudaKernelError();
};

template<typename DType>
Offload<DType>::Offload(int64_t bufferSize) {
    cudaHostAlloc(&errorArrayHost_, BUFFERSIZE_FACTOR * sizeof(float_t) * bufferSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&errorArrayDevice_, errorArrayHost_, 0);
    cudaHostAlloc(&indexArrayHost_, BUFFERSIZE_FACTOR * sizeof(uint64_t) * bufferSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&indexArrayDevice_, indexArrayHost_, 0);
    cudaHostAlloc(&inputContainerDataHost_, BUFFERSIZE_FACTOR * sizeof(DType) * bufferSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&inputContainerDataDevice_, inputContainerDataHost_, 0);
    cudaHostAlloc(&sumsHost_, BUFFERSIZE_FACTOR * sizeof(DType) * MAX_NUM_BLOCKS, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&sumsDevice_, sumsHost_, 0);
    cudaHostAlloc(&statesHost_, BUFFERSIZE_FACTOR * sizeof(curandState) * bufferSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&statesDevice_, statesHost_, 0);
    uniquePriorities_.reserve(bufferSize);
    priorityFrequencies_.reserve(bufferSize);
    result = std::vector<DType>(BUFFERSIZE_FACTOR * bufferSize);
    allocatedCapacity_ = BUFFERSIZE_FACTOR * bufferSize;
}

template<typename DType>
Offload<DType>::~Offload() {
    cudaFreeHost(errorArrayHost_);
    cudaFree(errorArrayDevice_);
    cudaFreeHost(indexArrayHost_);
    cudaFree(indexArrayDevice_);
    cudaFreeHost(inputContainerDataHost_);
    cudaFree(inputContainerDataDevice_);
    cudaFreeHost(statesHost_);
    cudaFree(statesDevice_);
    cudaFreeHost(sumsHost_);
    cudaFree(sumsDevice_);
}

template<typename DType>
template<class Container>
DType Offload<DType>::cumulative_sum(const Container &inputContainer, const int64_t parallelismSizeThreshold) {
    /*
   * Performs sum reduction on CUDA to compute cumulative sum of given inputContainer.
   * Uses CUDA only when parallelismSizeThreshold < inputContainer.size(). Else uses CPU and finds sum iteratively.
   *
   * Template Arguments
   * - Container: Must be an STL Container, hence must have implemented `size` method.
   * - DType: The datatype used for STL Container.
   */
    auto numElements = inputContainer.size();
    DType sumResultHost = 0;
    // If number of elements is less that `parallelismSizeThreshold`, find the sum on CPU.
    if (numElements < parallelismSizeThreshold) {
        for (int64_t index = 0; index != numElements; index++) {
            sumResultHost += inputContainer[index];
        }
        return sumResultHost;
    } else {
        fill_input_data(inputContainer, parallelismSizeThreshold);
    }
    // Identify number of CUDA blocks required to process the array.
    int numBlocks = numElements / NUM_THREADS;
    // If number of CUDA blocks is greater than `MAX_NUM_BLOCKS`,
    // slice the array via `cudaMemcpy` and recursively call
    // `cumulative_sum` to compute the sum.
    if (numBlocks > MAX_REDUCTION_BLOCKS) {
        int numElementsInSlice = NUM_THREADS * MAX_REDUCTION_BLOCKS;
        std::vector<DType> sliceVector(numElementsInSlice);
        DType totalSum = 0;
        bool enableParallelism = parallelismSizeThreshold < numElements;
#pragma omp parallel for if (enableParallelism) default(none)                                \
        firstprivate(numElementsInSlice, numElements, sliceVector, parallelismSizeThreshold) \
                shared(inputContainerDataHost_, totalSum)
        for (int index = 0; index < numElements; index += numElementsInSlice) {
            numElementsInSlice =
                    numElementsInSlice + index > numElements ? numElements - index : numElementsInSlice;
            if (numElementsInSlice > 0) {
                cudaMemcpy(&sliceVector[0],
                           &inputContainerDataHost_[index],
                           sizeof(DType) * numElementsInSlice,
                           cudaMemcpyHostToHost);
                auto sumTemp = cumulative_sum(sliceVector, parallelismSizeThreshold);
#pragma omp atomic update
                totalSum += sumTemp;
            }
        }
        return totalSum;
    }
    // Launch the CUDA Kernel.
    cumulative_sum_kernel<DType><<<numBlocks, NUM_THREADS>>>(sumsDevice_,
                                                             inputContainerDataDevice_,
                                                             numElements,
                                                             numBlocks);
    cudaDeviceSynchronize();
    cudaKernelError();
    for (int32_t index = 0; index < numBlocks; index++) {
        sumResultHost += sumsHost_[index];
    }
    return sumResultHost;
}

template<typename DType>
template<class Container>
void Offload<DType>::shuffle(const Container &inputContainer, const int64_t parallelismSizeThreshold) {
    /*
   * Shuffles the inputContainer in CUDA.
   *
   * Template Arguments
   * - Container: Must be an STL Container, hence must have implemented `size` method.
   */
    auto numElements = fill_input_data(inputContainer, parallelismSizeThreshold);
    shuffle_(numElements, parallelismSizeThreshold);
}

template<typename DType>
void Offload<DType>::generate_priority_seeds(DType cumulativeSum,
                                             int64_t parallelismSizeThreshold,
                                             uint64_t startPoint) {
    /*
   * Generates seeds with addition of random uniform error on CUDA.
   *
   * - Container: Must be an STL Container, hence must have implemented `size` method.
   * - DType: The datatype used for STL Container.
   */
    std::random_device rd;
    std::mt19937 generator(rd());
    size_t numElements = ceil(static_cast<float_t>(cumulativeSum));
    if (numElements > allocatedCapacity_){
        numElements = allocatedCapacity_;
    }
    bool enableParallelism = parallelismSizeThreshold < numElements;
    // Compute number of CUDA Blocks.
    auto numBlocksFloat = (float_t) numElements / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    // If more blocks than set by macros, call this function recursively.
    if (numBlocks > MAX_NUM_BLOCKS) {
        uint64_t numElementsInSlice = MAX_NUM_BLOCKS * NUM_THREADS;
        uint64_t numRounds = ceil(cumulativeSum / numElementsInSlice);
        float_t averageShareInCumulativeSum = cumulativeSum / numElements;
        std::uniform_int_distribution<uint64_t> randomIndexDistribution(0, numElements - 1);
#pragma omp parallel for if (enableParallelism) default(none) \
        firstprivate(numElementsInSlice, numElements, numRounds, averageShareInCumulativeSum, parallelismSizeThreshold)
        for (uint64_t startPoint_ = 0; startPoint_ < numRounds; startPoint_ += numElementsInSlice) {
            if (startPoint_ + numElementsInSlice > numElements) {
                numElementsInSlice = numElements - startPoint_;
            }
            float_t cumulativeSum_ = averageShareInCumulativeSum * static_cast<float_t>(numElementsInSlice);
            generate_priority_seeds(cumulativeSum_, parallelismSizeThreshold, startPoint_);
        }
        cudaMemcpy(&result[0], &inputContainerDataHost_[0], sizeof(DType) * numElements, cudaMemcpyHostToHost);
        return;
    }

    std::uniform_int_distribution<uint64_t> seedDistribution(INT64_MIN, INT64_MAX);
    auto seed = seedDistribution(generator);
    // Call to fill kernel to create array and add random error.
    array_index_fill_with_random_uniform_error_kernel<DType><<<numBlocks, NUM_THREADS>>>(inputContainerDataDevice_,
                                                                                         startPoint,
                                                                                         seed,
                                                                                         statesDevice_,
                                                                                         static_cast<size_t>(cumulativeSum));
    cudaDeviceSynchronize();
    cudaKernelError();
    // Call to shuffle function to shuffle the generated seeds.
    shuffle_(numElements, parallelismSizeThreshold);
}

template<typename DType>
template<class Container>
void Offload<DType>::arg_quantile_segment_indices(int64_t numSegments, const Container &inputContainer,
                                                  int64_t parallelismSizeThreshold) {

    auto numElements = fill_input_data(inputContainer, parallelismSizeThreshold);
    // Compute number of CUDA Blocks.
    auto numBlocksFloat = static_cast<float_t>(numElements) / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    if (numBlocks > MAX_NUM_BLOCKS) {
        uint64_t numElementsInSlice = NUM_THREADS * MAX_NUM_BLOCKS;
        bool enableParallelism = parallelismSizeThreshold < numSegments;
#pragma omp parallel for if (enableParallelism) default(none) \
        firstprivate(numElementsInSlice, numElements)         \
                shared(indexArrayDevice_)
        for (uint64_t startPoint_ = 0; startPoint_ < numElements; startPoint_ += numElementsInSlice) {
            if (startPoint_ + numElementsInSlice >= numElements) {
                numElementsInSlice = numElements - startPoint_;
            }
            int32_t newNumBlocks = ceil(static_cast<float_t>(numElementsInSlice) / NUM_THREADS);
            array_index_fill_kernel<<<newNumBlocks, NUM_THREADS>>>(indexArrayDevice_, startPoint_, numElements);
            cudaDeviceSynchronize();
            cudaKernelError();
        }
    } else {
        array_index_fill_kernel<<<numBlocks, NUM_THREADS>>>(indexArrayDevice_, 0, numElements);
        cudaDeviceSynchronize();
        cudaKernelError();
    }
    arg_mergesort(inputContainerDataHost_, indexArrayHost_, 0, numElements - 1, parallelismSizeThreshold);
    for (uint64_t index = 0; index < numElements; index++) {
        auto priority = inputContainerDataHost_[index];
        if (uniquePriorities_.empty() or uniquePriorities_.back() != priority) {
            uniquePriorities_.push_back(priority);
            priorityFrequencies_.push_back(1.0);
            continue;
        }
        priorityFrequencies_.back() += 1.0;
    }
    // Cumulative sun of the frequencies.
    auto cumulativeFrequencySum = cumulative_sum(priorityFrequencies_, parallelismSizeThreshold);
    // `segmentsInfo` will store quantile index as a pair of sorted indices, original indices.
    for (int64_t index = 1; index < numSegments + 1; index++) {
        int64_t quantileIndex = ceil(
                cumulativeFrequencySum * (static_cast<float_t>(index) / static_cast<float_t>(numSegments)));
        result[index - 1] = indexArrayHost_[quantileIndex - 1];
    }
    uniquePriorities_.clear();
    priorityFrequencies_.clear();
}

template<typename DType>
void Offload<DType>::reset() {
    {
#pragma omp parallel for default(none) shared(result)
        for (uint64_t index = 0; index < result.size(); index++) {
            result[index] = 0;
        }
    }
}

template<typename DType>
void Offload<DType>::shuffle_(size_t numElements,
                              int64_t parallelismSizeThreshold) {
    // Setup random device and generator.
    std::random_device rd;
    std::mt19937 generator(rd());
    // Compute number of CUDA Blocks.
    auto numBlocksFloat = (float_t) numElements / NUM_THREADS;
    uint32_t numBlocks = ceil(numBlocksFloat);
    execute_random_setup_kernel(numElements, 0, parallelismSizeThreshold);
    cudaKernelError();
    arg_mergesort(&errorArrayHost_[0],
                  &indexArrayHost_[0],
                  0, static_cast<int64_t>(numElements) - 1,
                  parallelismSizeThreshold);
    cudaKernelError();
    execute_shuffle_kernel(numElements, 0, parallelismSizeThreshold);
    cudaKernelError();
    if (numBlocks > MAX_NUM_BLOCKS) {
        std::uniform_int_distribution<uint64_t> randomIndexDistribution(0, numElements - 1);
        for (uint64_t round = 0; round < MAX_SHUFFLE_ROUNDS; round++) {
            auto randomIndex = randomIndexDistribution(generator);
            auto numElementsInSlice = numElements - randomIndex;
            execute_shuffle_kernel(numElementsInSlice,
                                   randomIndex,
                                   parallelismSizeThreshold);
            cudaKernelError();
        }
    }
    cudaMemcpy(&result[0],
               inputContainerDataHost_,
               sizeof(DType) * numElements,
               cudaMemcpyHostToHost);
}

template<typename DType>
void Offload<DType>::execute_random_setup_kernel(size_t numElements, uint64_t startPoint, int64_t parallelismSizeThreshold) {
    auto numBlocksFloat = (float_t) numElements / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    if (numBlocks > MAX_NUM_BLOCKS) {
        size_t numElementsInSlice = NUM_THREADS * MAX_NUM_BLOCKS;
        bool enableParallelism = parallelismSizeThreshold < numElements;
#pragma omp parallel for default(none) if (enableParallelism) \
        firstprivate(numElementsInSlice, numElements, startPoint, parallelismSizeThreshold)
        for (uint64_t indexPoint = startPoint; indexPoint < numElements; indexPoint += numElementsInSlice) {
            if (indexPoint + numElementsInSlice > numElements) {
                numElementsInSlice = numElements - indexPoint;
            }
            execute_random_setup_kernel(numElementsInSlice,
                                        indexPoint,
                                        parallelismSizeThreshold);
        }
        return;
    }
    // Setup random device and generator.
    std::random_device rd;
    std::mt19937 generator(rd());
    // Uniform Int Distribution to generate seed.
    std::uniform_int_distribution<uint64_t> seedDistribution(INT64_MIN, INT64_MAX);
    auto seed = seedDistribution(generator);
    // Call CUDA kernel to set up random.
    random_setup_kernel<<<numBlocks, NUM_THREADS>>>(&indexArrayDevice_[startPoint],
                                                    &errorArrayDevice_[startPoint],
                                                    seed,
                                                    &statesDevice_[startPoint],
                                                    startPoint,
                                                    numElements);
    cudaDeviceSynchronize();
}

template<typename DType>
void Offload<DType>::execute_shuffle_kernel(size_t numElements, uint64_t startPoint, int64_t parallelismSizeThreshold) {
    auto numBlocksFloat = (float_t) numElements / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    if (numBlocks > MAX_NUM_BLOCKS) {
        size_t numElementsInSlice = NUM_THREADS * MAX_NUM_BLOCKS;
        bool enableParallelism = parallelismSizeThreshold < numElements;
#pragma omp parallel for default(none) if (enableParallelism) \
        firstprivate(numElementsInSlice, numElements, startPoint, parallelismSizeThreshold)
        for (uint64_t indexPoint = startPoint; indexPoint < numElements; indexPoint += numElementsInSlice) {
            if (indexPoint + numElementsInSlice > numElements) {
                numElementsInSlice = numElements - indexPoint;
            }
            execute_shuffle_kernel(numElementsInSlice, indexPoint, parallelismSizeThreshold);
        }
        return;
    }
    shuffle_kernel<DType><<<numBlocks, NUM_THREADS>>>(inputContainerDataDevice_,
                                                      indexArrayDevice_,
                                                      numElements, startPoint);
    cudaDeviceSynchronize();
}


template<typename DType>
template<class Container>
size_t Offload<DType>::fill_input_data(Container &inputContainer, int64_t parallelismSizeThreshold) {
    size_t numElements = inputContainer.size();
    if (match_vector_type<Container>) {
        cudaMemcpy(&inputContainerDataHost_[0], &inputContainer[0], sizeof(DType) * numElements, cudaMemcpyHostToHost);
    } else {
        // If number of elements is greater that `parallelismSizeThreshold`, find the sum on CUDA.
        bool enableParallelism = parallelismSizeThreshold < numElements;
        {
            // Parallel region moves inputContainer's data to a C array parallely.
#pragma omp parallel for if (enableParallelism) default(none) \
        firstprivate(inputContainer, numElements)             \
                shared(inputContainerDataHost_)
            for (int64_t index = 0; index < numElements; index++) {
                inputContainerDataHost_[index] = inputContainer[index];
            }
        }
    }
    return numElements;
}

template<typename DType>
void Offload<DType>::cudaKernelError() {
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::string cudaErrorMessage = cudaGetErrorString(status);
        throw std::runtime_error("CUDA Kernel failed: " + cudaErrorMessage);
    }
}

template<typename DType>
__device__ void cumulative_sum_device_fn(DType *sum, const DType *vector) {
    __shared__ DType sharedData[NUM_THREADS];
    auto tid = threadIdx.x;
    sharedData[tid] = vector[tid];
    for (unsigned int s = (blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum[blockIdx.x] = sharedData[0];
    }
}

template<typename DType>
__global__ void cumulative_sum_kernel(DType *sums,
                                      const DType *data,
                                      const size_t size,
                                      const int64_t numBlocks) {
    unsigned int forwardIndex = (numBlocks * blockIdx.x) + threadIdx.x;
    if (forwardIndex < size) {
        G_sharedData[blockIdx.x][threadIdx.x] = data[forwardIndex];
        __syncthreads();
        cumulative_sum_device_fn<DType>(sums, G_sharedData[blockIdx.x]);
    } else {
        G_sharedData[blockIdx.x][threadIdx.x] = 0;
    }
}

template<typename DType>
__global__ void random_setup_kernel(uint64_t *sortedIndices,
                                    DType *sortedRandomValues,
                                    uint64_t seed,
                                    curandState *cuRandStates,
                                    uint64_t startPoint,
                                    size_t size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        curand_init(seed, tid, 0, &cuRandStates[tid]);
        sortedRandomValues[tid] = curand_uniform(&cuRandStates[tid]);
        sortedIndices[tid] = startPoint + tid;
    }
}

template<typename DType>
__global__ void shuffle_kernel(DType *shuffle, const uint64_t *shuffledIndices, size_t size, uint64_t startPoint) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        tid += startPoint;
        float_t currentElement = shuffle[shuffledIndices[tid]];
        shuffle[shuffledIndices[tid]] = shuffle[tid];
        shuffle[tid] = currentElement;
        __syncthreads();
    }
}

template<typename DType>
__global__ void array_index_fill_with_random_uniform_error_kernel(DType *fillArray,
                                                                  uint64_t start,
                                                                  uint64_t seed,
                                                                  curandState *cuRandStates,
                                                                  size_t size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        tid += start;
        curand_init(seed, tid, 0, &cuRandStates[tid]);
        fillArray[tid] = curand_uniform(&cuRandStates[tid]) + static_cast<DType>(tid);
    }
    __syncthreads();
}

template<typename DType>
__global__ void array_index_fill_kernel(DType *fillArray,
                                        uint64_t start,
                                        size_t size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        fillArray[tid] = start + tid;
    }
}

#endif//__CUDA_AVAILABLE__
#endif//RLPACK_BINARIES_MEMORY_UTILS_CUDAOFFLOAD_CUH_