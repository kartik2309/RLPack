//
// Created by Kartik Rajeshwaran on 2022-11-01.
//


#ifdef __CUDA_AVAILABLE__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <random>
#include <vector>

#include "../../utils/ops/arg_mergesort.cuh"

#define MAX_NUM_BLOCKS 8
#define MAX_REDUCTION_BLOCKS 2
#define NUM_THREADS 1024
#define MAX_SHUFFLE_ROUNDS 1024
#define INCREASE_SHUFFLE_ROUNDS_THRESHOLD 1048576
#define MAX_SHUFFLE_ROUNDS_THRESHOLD 8192

//template<typename DType>
__shared__ float_t G_sharedData[MAX_NUM_BLOCKS][NUM_THREADS];

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
                                    size_t size);

template<typename DType>
__global__ void shuffle_kernel(DType *shuffle, const uint64_t *shuffledIndices, size_t size);

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

    Offload(int64_t bufferSize, int32_t batchSize);
    ~Offload();

    template<class Container>
    DType cumulative_sum(const Container &inputContainer, int64_t parallelismSizeThreshold);

    template<class Container>
    Container shuffle(const Container &inputContainer, int64_t parallelismSizeThreshold);

    template<class Container>
    Container generate_priority_seeds(DType cumulativeSum,
                                      int64_t parallelismSizeThreshold,
                                      uint64_t startPoint = 0.0);

    template<class Container>
    Container arg_quantile_segment_indices(int64_t numSegments,
                                           const Container &inputContainer,
                                           int64_t parallelismSizeThreshold);

private:
    float_t *errorArrayHost_, *errorArrayDevice_;
    uint64_t *indexArrayHost_, *indexArrayDevice_;
    DType *inputContainerDataHost_, *inputContainerDataDevice_;
    std::vector<DType> uniquePriorities_;
    std::vector<int64_t> priorityFrequencies_;
    std::vector<DType> toShuffleVector_;
};

template<typename DType>
Offload<DType>::Offload(int64_t bufferSize, int32_t batchSize) {
    errorArrayHost_ = new float_t[bufferSize];
    cudaMalloc(&errorArrayDevice_, sizeof(float_t) * bufferSize);
    indexArrayHost_ = new uint64_t[bufferSize];
    cudaMalloc(&indexArrayDevice_, sizeof(uint64_t) * bufferSize);
    inputContainerDataHost_ = new DType[bufferSize];
    cudaMalloc(&inputContainerDataDevice_, sizeof(DType) * bufferSize);
    uniquePriorities_.reserve(bufferSize);
    priorityFrequencies_.reserve(bufferSize);
    result = std::vector<DType>(bufferSize);
    toShuffleVector_ = std::vector<DType>(batchSize);
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
    }
    // If number of elements is greater that `parallelismSizeThreshold`, find the sum on CUDA.
    auto *inputContainerData = new DType[numElements];
    bool enableParallelism = parallelismSizeThreshold < numElements;
    {
        // Parallel region moves inputContainer's data to a C array parallely.
#pragma omp parallel for if (enableParallelism) default(none) firstprivate(inputContainer) shared(inputContainerData)
        for (int64_t index = 0; index < numElements; index++) {
            inputContainerData[index] = inputContainer[index];
        }
    }
    // Identify number of CUDA blocks required to process the array.
    int numBlocks = numElements / NUM_THREADS;
    // If number of CUDA blocks is greater than `MAX_NUM_BLOCKS`, slice the array via `memcpy` and recursively call
    // `cumulative_sum` to compute the sum.
    if (numBlocks > MAX_REDUCTION_BLOCKS) {
        int totalElementsInSlice = NUM_THREADS * MAX_REDUCTION_BLOCKS;
        DType totalSum = 0;
        for (int index = 0; index <= numElements; index += totalElementsInSlice) {
            int actualSize =
                    totalElementsInSlice + index > numElements ? numElements - index
                                                               : totalElementsInSlice;
            if (actualSize > 0) {
                std::vector<DType> sliceVector(actualSize);
                memcpy(&sliceVector[0], &inputContainerData[index], sizeof(DType) * actualSize);
                totalSum += cumulative_sum(sliceVector, parallelismSizeThreshold);
            }
        }
        return totalSum;
    }
    DType *inputContainerDevice, *sumsDevice, *sumsHost;
    sumsHost = new DType[numBlocks];
    // Allocate memory in CUDA for container data and sums to store sum from each CUDA block.
    cudaMalloc(&inputContainerDevice, sizeof(DType) * numElements);
    cudaMalloc(&sumsDevice, sizeof(DType) * numBlocks);
    // Move container data to CUDA.
    cudaMemcpy(
            inputContainerDevice,
            inputContainerData,
            sizeof(DType) * numElements,
            cudaMemcpyHostToDevice);
    // Launch the CUDA Kernel.
    cumulative_sum_kernel<DType><<<numBlocks, NUM_THREADS>>>(sumsDevice, inputContainerDevice, numElements, numBlocks);
    cudaDeviceSynchronize();
    // Move back the sums from each CUDA block to Host.
    cudaMemcpy(sumsHost, sumsDevice, sizeof(DType) * numBlocks, cudaMemcpyDeviceToHost);
    // Find the sum.
    for (int i = 0; i < numBlocks; i++) {
        sumResultHost += sumsHost[i];
    }
    // Deallocate the memory and return the control.
    cudaFree(inputContainerDevice);
    cudaFree(sumsDevice);
    delete[] sumsHost;
    return sumResultHost;
}

template<typename DType>
template<class Container>
Container Offload<DType>::shuffle(const Container &inputContainer, const int64_t parallelismSizeThreshold) {
    /*
   * Shuffles the inputContainer in CUDA.
   *
   * Template Arguments
   * - Container: Must be an STL Container, hence must have implemented `size` method.
   * - DType: The datatype used for STL Container.
   */
    // Setup random device and generator.
    std::random_device rd;
    std::mt19937 generator(rd());
    auto numElements = inputContainer.size();
    // Container to store shuffled results.
    Container shuffledContainer(numElements);
    // C Array allocation to store data underlying the container.
    auto *inputContainerData = new DType[numElements];
    bool enableParallelism = parallelismSizeThreshold < numElements;
    {
        // Parallel region moves inputContainer's data to a C array parallely.
#pragma omp parallel for if (enableParallelism) default(none) firstprivate(inputContainer) shared(inputContainerData)
        for (int64_t index = 0; index < numElements; index++) {
            inputContainerData[index] = inputContainer[index];
        }
    }
    // Compute number of CUDA Blocks.
    auto numBlocksFloat = (float_t) numElements / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    // If number of CUDA blocks are higher than set MAX_NUM_BLOCKS, iteratively shuffle.
    if (numBlocks > MAX_NUM_BLOCKS) {
        std::uniform_int_distribution<uint64_t> indexDistribution(0, numElements - 1);
        size_t maxNumElementsInSlice = numElements / MAX_SHUFFLE_ROUNDS;
        int32_t maxRounds = MAX_SHUFFLE_ROUNDS;
        if (numElements > INCREASE_SHUFFLE_ROUNDS_THRESHOLD) {
            int32_t maxRoundsMultiplier = ceil((float_t) numElements / INCREASE_SHUFFLE_ROUNDS_THRESHOLD);
            maxRounds *= maxRoundsMultiplier;
            if (maxRounds > MAX_SHUFFLE_ROUNDS_THRESHOLD) {
                maxRounds = MAX_SHUFFLE_ROUNDS_THRESHOLD;
            }
            maxNumElementsInSlice = numElements / maxRounds;
        }
        bool traversalComplete = false;
        {
#pragma omp parallel for default(none) shared(inputContainerData) schedule(static)
            for (uint64_t round = 0; round < maxRounds; round++) {
                size_t numElementsInSlice = maxNumElementsInSlice;
                uint64_t randomStartIndex = 0;
                if (traversalComplete) {
                    randomStartIndex = indexDistribution(generator);
                } else {
                    randomStartIndex = round * numElementsInSlice;
                    if (randomStartIndex >= numElements) {
                        traversalComplete = true;
                        randomStartIndex = indexDistribution(generator);
                    }
                }
                if (randomStartIndex + numElementsInSlice > numElements) {
                    numElementsInSlice = numElements - randomStartIndex;
                }
                std::vector<DType> inputContainerVectorSlice(numElementsInSlice);
                memcpy(&inputContainerVectorSlice[0], &inputContainerData[randomStartIndex],
                       sizeof(DType) * numElementsInSlice);
                auto shuffledVector = shuffle<std::vector<DType>>(inputContainerVectorSlice,
                                                                  parallelismSizeThreshold);
                memcpy(&inputContainerData[randomStartIndex], &shuffledVector[0],
                       sizeof(DType) * numElementsInSlice);
            }
        }
        {
            // Parallel region moves inputContainerData's data to shuffledContainer parallely.
#pragma omp parallel for if (enableParallelism) default(none) firstprivate(inputContainerData) shared(shuffledContainer)
            for (int64_t index = 0; index < numElements; index++) {
                shuffledContainer[index] = inputContainerData[index];
            }
        }
        return shuffledContainer;
    }
    // Uniform Int Distribution to generate seed.
    std::uniform_int_distribution<uint64_t> seedDistribution(INT64_MIN, INT64_MAX);
    auto seed = seedDistribution(generator);
    // Dynamically allocate an array of numElements of curandState for CUDA.
    auto statesHost = new curandState[numElements];
    curandState *statesDevice;
    // Sorted indices and random values for Device and Host.
    uint64_t *sortedIndicesDevice, *sortedIndicesHost;
    DType *sortedRandomValuesDevice, *sortedRandomValuesHost, *inputContainerDataDevice;
    // Allocate memory for indices array in Host and CUDA.
    sortedIndicesHost = new uint64_t[numElements];
    cudaMalloc(&sortedIndicesDevice, sizeof(uint64_t) * numElements);
    // Allocate memory for random values array in Host and CUDA.
    sortedRandomValuesHost = new DType[numElements];
    cudaMalloc(&sortedRandomValuesDevice, sizeof(DType) * numElements);
    // Allocate memory for states array in CUDA.
    cudaMalloc(&statesDevice, sizeof(curandState) * numElements);
    cudaMemcpy(statesDevice, statesHost, sizeof(curandState) * numElements, cudaMemcpyHostToDevice);
    // Call CUDA kernel to set up random. This fills  `sortedIndicesDevice` with indices and `sortedRandomValuesDevice`
    // with the corresponding random value.
    random_setup_kernel<<<numBlocks, NUM_THREADS>>>(sortedIndicesDevice, sortedRandomValuesDevice, seed, statesDevice,
                                                    numElements);
    // Copy random values and indices to Host.
    cudaMemcpy(sortedIndicesHost, sortedIndicesDevice, sizeof(uint64_t) * numElements, cudaMemcpyDeviceToHost);
    cudaMemcpy(sortedRandomValuesHost, sortedRandomValuesDevice, sizeof(float_t) * numElements, cudaMemcpyDeviceToHost);
    // Perform Merge sort with key on host.
    arg_mergesort(sortedRandomValuesHost, sortedIndicesHost, 0, numElements - 1, parallelismSizeThreshold);
    // Move the sorted indices to CUDA.
    cudaMemcpy(sortedIndicesDevice, sortedIndicesHost, sizeof(uint64_t) * numElements, cudaMemcpyHostToDevice);
    // Allocate memory for inputContainerDataDevice, inputContainerData to be moved to CUDA. This will have shuffled data.
    cudaMalloc(&inputContainerDataDevice, sizeof(DType) * numElements);
    // Move inputContainerData to CUDA.
    cudaMemcpy(inputContainerDataDevice, inputContainerData, sizeof(DType) * numElements, cudaMemcpyHostToDevice);
    // Call shuffling kernel.
    shuffle_kernel<DType><<<numBlocks, NUM_THREADS>>>(inputContainerDataDevice, sortedIndicesDevice, numElements);
    // Move shuffled data back to Host.
    cudaMemcpy(inputContainerData, inputContainerDataDevice, sizeof(DType) * numElements, cudaMemcpyDeviceToHost);
    {
        // Parallel region moves inputContainerData's data to shuffledContainer parallely.
#pragma omp parallel for if (enableParallelism) default(none) firstprivate(inputContainerData) shared(shuffledContainer)
        for (int64_t index = 0; index < numElements; index++) {
            shuffledContainer[index] = inputContainerData[index];
        }
    }
    // Deallocate memory in CUDA and Host.
    cudaFree(statesDevice);
    cudaFree(sortedRandomValuesDevice);
    cudaFree(sortedIndicesDevice);
    cudaFree(inputContainerDataDevice);
    delete[] statesHost;
    delete[] sortedRandomValuesHost;
    delete[] sortedIndicesHost;
    delete[] inputContainerData;
    return shuffledContainer;
}

template<typename DType>
template<class Container>
Container Offload<DType>::generate_priority_seeds(DType cumulativeSum,
                                                  int64_t parallelismSizeThreshold,
                                                  uint64_t startPoint) {
    /*
   * Generates seeds with addition of random uniform error on CUDA.
   *
   * - Container: Must be an STL Container, hence must have implemented `size` method.
   * - DType: The datatype used for STL Container.
   */
    size_t numElements = ceil(static_cast<float_t>(cumulativeSum));
    auto *fillArrayHost = new DType[numElements];
    Container prioritySeeds(numElements);
    bool enableParallelism = parallelismSizeThreshold < numElements;
    // Compute number of CUDA Blocks.
    auto numBlocksFloat = (float_t) numElements / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    // If more blocks than set by macros, call this function recursively.
    if (numBlocks > MAX_NUM_BLOCKS) {
        uint64_t totalElementsInSlice = NUM_THREADS * MAX_NUM_BLOCKS,
                 numRounds = static_cast<int64_t>(ceil(numElements / totalElementsInSlice));
        DType cumulativeSumPerSlice = cumulativeSum / numRounds;
        {
#pragma omp parallel for if (enableParallelism) default(none)                                          \
        firstprivate(cumulativeSumPerSlice, totalElementsInSlice, numRounds, parallelismSizeThreshold) \
                shared(fillArrayHost)
            for (uint64_t startPoint_ = 0; startPoint_ < numElements; startPoint_ += totalElementsInSlice) {
                if (startPoint_ + totalElementsInSlice >= numElements) {
                    totalElementsInSlice = numElements - startPoint_;
                }
                auto prioritySeedsSlice = generate_priority_seeds<std::vector<DType>>(
                        cumulativeSumPerSlice,
                        parallelismSizeThreshold,
                        startPoint_);
                memcpy(&fillArrayHost[startPoint_],
                       &prioritySeedsSlice[0],
                       sizeof(DType) * totalElementsInSlice);
            }
        }
        {
#pragma omp parallel for if (enableParallelism) default(none) shared(prioritySeeds) firstprivate(fillArrayHost)
            for (uint64_t index = 0; index < numElements; index++) {
                prioritySeeds[index] = fillArrayHost[index];
            }
        }
        return prioritySeeds;
    }
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<uint64_t> seedDistribution(INT64_MIN, INT64_MAX);
    auto seed = seedDistribution(generator);
    // Dynamically allocate fillArray for CUDA.
    DType *fillArrayDevice;
    cudaMalloc(&fillArrayDevice, sizeof(DType) * numElements);
    // Dynamically allocate an array of numElements of curandState for CUDA.
    auto statesHost = new curandState[numElements];
    curandState *statesDevice;
    cudaMalloc(&statesDevice, sizeof(curandState) * numElements);
    // Copy random state objects from Host to GPU.
    cudaMemcpy(statesDevice,
               statesHost,
               sizeof(curandState) * numElements,
               cudaMemcpyHostToDevice);
    // Call to fill kernel to create array and add random error.
    array_index_fill_with_random_uniform_error_kernel<DType><<<numBlocks, NUM_THREADS>>>(fillArrayDevice,
                                                                                         startPoint,
                                                                                         seed,
                                                                                         statesDevice,
                                                                                         numElements);
    // Copy the results from GPU to Host
    cudaMemcpy(fillArrayHost,
               fillArrayDevice,
               sizeof(DType) * numElements,
               cudaMemcpyDeviceToHost);
    // Vector Allocation and transfer of generated seeds to the vector
    std::vector<DType> fillArrayVector(numElements);
    memmove(fillArrayVector.data(), fillArrayHost, sizeof(DType) * numElements);
    // Call to shuffle function to shuffle the generated seeds.
    auto prioritySeedsVector = shuffle(fillArrayVector, parallelismSizeThreshold);
    {
        // Parallel region to transfer of shuffled seeds to desired Container specified as per template.
#pragma omp parallel for if (enableParallelism) default(none) shared(prioritySeeds) firstprivate(prioritySeedsVector)
        for (uint64_t index = 0; index < numElements; index++) {
            prioritySeeds[index] = prioritySeedsVector[index];
        }
    }
    // Deallocate memory from GPU and Host
    cudaFree(fillArrayDevice);
    delete[] fillArrayHost;
    return prioritySeeds;
}

template<typename DType>
template<class Container>
Container Offload<DType>::arg_quantile_segment_indices(int64_t numSegments, const Container &inputContainer,
                                                       int64_t parallelismSizeThreshold) {
    size_t numElements = inputContainer.size();
    std::vector<DType> uniquePriorities, priorityFrequencies;
    Container segmentsInfo(numSegments);
    bool enableParallelism = parallelismSizeThreshold < numElements;
    auto inputContainerDataHost = new DType[numElements];
    {
        // Parallel region moves inputContainer's data to a C array parallely.
#pragma omp parallel for if (enableParallelism) default(none) firstprivate(inputContainer) shared(inputContainerDataHost)
        for (int64_t index = 0; index < numElements; index++) {
            inputContainerDataHost[index] = inputContainer[index];
        }
    }
    // Compute number of CUDA Blocks.
    auto numBlocksFloat = static_cast<float_t>(numElements) / NUM_THREADS;
    int32_t numBlocks = ceil(numBlocksFloat);
    auto sortedIndicesHost = new uint64_t[numElements];
    if (numBlocks > MAX_NUM_BLOCKS) {
        uint64_t totalElementsInSlice = NUM_THREADS * MAX_NUM_BLOCKS;
        for (uint64_t startPoint_ = 0; startPoint_ < numElements; startPoint_ += totalElementsInSlice) {
            if (startPoint_ + totalElementsInSlice >= numElements) {
                totalElementsInSlice = numElements - startPoint_;
            }
            uint64_t *sortedIndicesDevice;
            cudaMalloc(&sortedIndicesDevice, sizeof(uint64_t) * totalElementsInSlice);
            int32_t newNumBlocks = ceil(static_cast<float_t>(totalElementsInSlice) / NUM_THREADS);
            array_index_fill_kernel<<<newNumBlocks, NUM_THREADS>>>(sortedIndicesDevice, startPoint_, numElements);
            cudaMemcpy(&sortedIndicesHost[startPoint_],
                       &sortedIndicesDevice[0],
                       sizeof(uint64_t) * totalElementsInSlice,
                       cudaMemcpyDeviceToHost);
            cudaFree(sortedIndicesDevice);
        }
    } else {
        uint64_t *sortedIndicesDevice;
        cudaMalloc(&sortedIndicesDevice, sizeof(uint64_t) * numElements);
        array_index_fill_kernel<<<numBlocks, NUM_THREADS>>>(sortedIndicesDevice, 0, numElements);
        cudaMemcpy(sortedIndicesHost,
                   sortedIndicesDevice,
                   sizeof(uint64_t) * numElements,
                   cudaMemcpyDeviceToHost);
        cudaFree(sortedIndicesDevice);
    }
    arg_mergesort(inputContainerDataHost, sortedIndicesHost, 0, numElements - 1, parallelismSizeThreshold);
    for (uint64_t index = 0; index < numElements; index++) {
        auto priority = inputContainerDataHost[index];
        if (uniquePriorities.empty() or uniquePriorities.back() != priority) {
            uniquePriorities.push_back(priority);
            priorityFrequencies.push_back(1.0);
            continue;
        }
        priorityFrequencies.back() += 1.0;
    }

    // Cumulative sun of the frequencies.
    auto cumulativeFrequencySum = cumulative_sum(priorityFrequencies, parallelismSizeThreshold);
    // `segmentsInfo` will store quantile index as a pair of sorted indices, original indices.
    for (int64_t index = 1; index < numSegments + 1; index++) {
        int64_t quantileIndex = ceil(
                cumulativeFrequencySum * (static_cast<float_t>(index) / static_cast<float_t>(numSegments)));
        segmentsInfo[index - 1] = sortedIndicesHost[quantileIndex - 1];
    }
    delete[] inputContainerDataHost;
    delete[] sortedIndicesHost;
    return segmentsInfo;
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
                                    size_t size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &cuRandStates[tid]);
    if (tid < size) {
        sortedRandomValues[tid] = curand_uniform(&cuRandStates[tid]);
        sortedIndices[tid] = tid;
    }
    __syncthreads();
}

template<typename DType>
__global__ void shuffle_kernel(DType *shuffle, const uint64_t *shuffledIndices, size_t size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float_t currentElement = shuffle[shuffledIndices[tid]];
        shuffle[shuffledIndices[tid]] = shuffle[tid];
        shuffle[tid] = currentElement;
    }
}

template<typename DType>
__global__ void array_index_fill_with_random_uniform_error_kernel(DType *fillArray,
                                                                  uint64_t start,
                                                                  uint64_t seed,
                                                                  curandState *cuRandStates,
                                                                  size_t size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &cuRandStates[tid]);

    if (tid < size) {
        fillArray[tid] = static_cast<DType>(start) + curand_uniform(&cuRandStates[tid]) + static_cast<DType>(tid);
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
