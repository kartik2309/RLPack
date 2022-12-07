
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
 * @defgroup cuda_group cuda
 * @brief CUDA optimized implementation for Offload
 * @{
 */

//! Maximum number of CUDA blocks that can be launched.
#define MAX_NUM_BLOCKS 256
//! Maximum number of CUDA blocks that can be launched for reduction.
#define MAX_REDUCTION_BLOCKS 2
//! Maximum number of threads per CUDA Blocks.
#define NUM_THREADS 1024
//! Maximum number of rounds for shuffling. This is used in Offload::shuffle when all input elements cannot fit in GPU.
#define MAX_SHUFFLE_ROUNDS 1024
//! Maximum shared memory blocks (as per 16KB limitation)
#define MAX_SHARED_MEMORY_BLOCKS 8
//! Factor by which buffer size is multiplied to allocate the memory in GPU and CPU.
#define BUFFERSIZE_FACTOR 7

//! Global shared memory for CUDA for data sharing between blocks.
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

/*!
 * @brief Template Offload class for GPU with CUDA optimized kernels with OpenMP optimized routines for host
 * operations. This class uses pinned CUDA memory based implementations.
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
    //! Float pointer to an array to store random errors in Host.
    float_t *errorArrayHost_;
    //! Float pointer to an array to store random errors in GPU.
    float_t *errorArrayDevice_;
    //! Int64 pointer to an array to store indices in Host.
    uint64_t *indexArrayHost_;
    //! Int64 pointer to an array to store indices in GPU.
    uint64_t *indexArrayDevice_;
    //! DType pointer to an array to store input container's data to be processed in Host.
    DType *inputContainerDataHost_;
    //! DType pointer to an array to store input container's data to be processed in GPU.
    DType *inputContainerDataDevice_;
    //! DType pointer to an array to store intermediate sums in Host.
    DType *sumsHost_;
    //! DType pointer to an array to store intermediate sums in GPU.
    DType *sumsDevice_;
    //! curandState pointer to an array to store random states for CUDA kernel random generator in Host.
    curandState *statesHost_;
    //! curandState pointer to an array to store random states for CUDA kernel random generator in GPU.
    curandState *statesDevice_;
    //! The Vector to store unique priorities. This is used in Offload::arg_quantile_segment_indices method.
    std::vector<DType> uniquePriorities_;
    //! The Vector to store intermediate values to be shuffled.
    std::vector<DType> toShuffleVector_;
    //! The Vector to store frequencies of each priority. This is used in Offload::arg_quantile_segment_indices method.
    std::vector<int64_t> priorityFrequencies_;
    //! Indicating the total capacity to be allocated in Host and CUDA.
    uint64_t allocatedCapacity_;

    void shuffle_(size_t numElements, int64_t parallelismSizeThreshold);

    void execute_random_setup_kernel(size_t numElements, uint64_t startPoint, int64_t parallelismSizeThreshold);

    void execute_shuffle_kernel(size_t numElements, uint64_t startPoint, int64_t parallelismSizeThreshold);

    template<class Container>
    size_t fill_input_data(Container &inputContainer, int64_t parallelismSizeThreshold);

    void cudaKernelError();
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
     * Constructor for Offload. Dynamically allocates required memory in Host and GPU and initialises necessary variables.
     */
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
    /*!
     * Destructor for Offload. De-allocated all dynamically allocated memory in Host and GPU
     */
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
    /*!
     * Performs sum reduction on CUDA to compute cumulative sum of given inputContainer.
     * Uses CUDA only when parallelismSizeThreshold < inputContainer.size(). Else uses CPU and finds sum iteratively.
     *
     * Template Arguments
     * - Container: Must be an STL Container initialised with DType, hence must have implemented `size` method.
     *
     * @param inputContainer : The input container for which sum has to be computed.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond with
     * CUDA Kernels are to be used.
     * @return The cumulative sum.
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
    /*!
     * This template method shuffles the given input container. Results are flushed into Offload::shuffle.
     *
     *
     * Template Arguments
     * - Container: Must be an STL Container initialised with DType, hence must have implemented `size` method.
     *
     * @param inputContainer : The input container which has to be shuffled.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP routines are to be used
     * with CUDA kernels.
     */
    auto numElements = fill_input_data(inputContainer, parallelismSizeThreshold);
    shuffle_(numElements, parallelismSizeThreshold);
}

template<typename DType>
void Offload<DType>::generate_priority_seeds(DType cumulativeSum,
                                             int64_t parallelismSizeThreshold,
                                             uint64_t startPoint) {
    /*!
     * This method generates the priority seeds. This is used by C_Memory::sample when using
     * proportional prioritization strategy. This method generates seeds between arguments
     * startPoint and cumulativeSum. Results are flushed into Offload::result.
     *
     * @param cumulativeSum : The cumulative sum upto which the seeds are to be generated.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP routines are to be used
     * with CUDA kernels.
     * @param startPoint : The start point, i.e. the smallest value of the generated seeds.
     */
    std::random_device rd;
    std::mt19937 generator(rd());
    size_t numElements = ceil(static_cast<float_t>(cumulativeSum));
    if (numElements > allocatedCapacity_) {
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
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP routines are to be used
     * with CUDA kernels.
     */
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

template<typename DType>
void Offload<DType>::shuffle_(size_t numElements,
                              int64_t parallelismSizeThreshold) {
    /*!
     * Helper method for Offload::shuffle in CUDA. This method invokes necessary methods to execute CUDA Kernels
     * required to obtain the results. This fetches the data from Offload::result.
     *
     * @param numElements : Number of elements in the to be shuffled based on number of elements loaded
     * in Offload::result.
     * @param parallelismSizeThreshold : The threshold size of Offload::result beyond which OpenMP routines are
     * to be used with CUDA kernels.
     */
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
void Offload<DType>::execute_random_setup_kernel(size_t numElements,
                                                 uint64_t startPoint,
                                                 int64_t parallelismSizeThreshold) {
    /*!
     * Helper method for Offload::shuffle_ in CUDA. This method executes the CUDA Kernels required to
     * set up the random states and errors to help shuffle the container.
     *
     * @param numElements : Number of elements in the to be shuffled based on number of elements loaded
     * in Offload::result.
     * @param startPoint : The start point from which CUDA kernel is supposed to full the error and index arrays
     * (Offload::errorArrayDevice_, Offload::indexArraydevice_). This is used for recursion when we have limited
     * number of blocks.
     * @param parallelismSizeThreshold : The threshold size of Offload::result beyond which OpenMP routines are
     * to be used with CUDA kernels.
     */
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
void Offload<DType>::execute_shuffle_kernel(size_t numElements,
                                            uint64_t startPoint,
                                            int64_t parallelismSizeThreshold) {
    /*!
     * Helper method for Offload::shuffle_ in CUDA. This method executes the CUDA Kernels required to
     * set up the perform final shuffle
     *
     * @param numElements : Number of elements in the to be shuffled based on number of elements loaded
     * in Offload::result.
     * @param startPoint : The start point from which CUDA kernel is supposed to full the error and index arrays
     * (Offload::errorArrayDevice_, Offload::indexArraydevice_). This is used for recursion when we have limited
     * number of blocks.
     * @param parallelismSizeThreshold : The threshold size of Offload::result beyond which OpenMP routines are
     * to be used with CUDA kernels.
     */
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
    /*!
     * Helper method for all methods in Offload with CUDA. This class moves the data from STL container to the raw
     * pointer array. It is moved to Offload::inputContainerDataHost_.
     * If `inputContainer` is a vector, memory block is copied and is efficient. In other cases (non-contingous
     * storage STLs), iterative copy is done with OpenMP routine.
     *
     *
     * Template Arguments
     * - Container: Must be an STL Container initialised with DType, hence must have implemented `size` method.
     *
     * @param inputContainer : The input container for which sum has to be computed.
     * @param parallelismSizeThreshold : The threshold size of inputContainer beyond which OpenMP
     * parallelized routines are to be used.
     * @return The size of input (`inputContainer`).
     */
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
    /*
     * CUDA method to raise errors for silent errors occurred during CUDA operations.
     */
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::string cudaErrorMessage = cudaGetErrorString(status);
        throw std::runtime_error("CUDA Kernel failed: " + cudaErrorMessage);
    }
}

template<typename DType>
__device__ void cumulative_sum_device_fn(DType *sum, const DType *vector) {
    /*!
     * CUDA Device to find cumulative sum.
     *
     * @param sum : Pointer to array in which result is to be accumulated.
     * @param vector : Pointer to array for which sum has to be calculated.
     */
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
    /*!
     * CUDA Kernel to find cumulative sum.
     *
     * @param sums : Pointer to array in which result is to be accumulated.
     * @param data : Pointer to array for which sum has to be calculated.
     * @param size : The size of the data up to which sum is to be computed.
     * @param numBlocks : Number of blocks that have been launched for this kernel.
     */
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
    /*!
     * CUDA Kernel to setup random error and their indices. Random errors are generated as per uniform distribution.
     *
     * @param sortedIndices : Pointer to array to which is to be accumulated with indices for random values.
     * @param sortedRandomValues : Pointer to array which is to be accumulated with random values.
     * @param seed : The seed value to initialize curand.
     * @param cuRandStates : Pointer to array of cuRandStates objects.
     * @param startPoint : Start point, only beyond which `sortedIndices` is filled.
     * @param size : The size of the `sortedIndices` and `sortedRandomValues` beyond which array is not filled.
     */
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        curand_init(seed, tid, 0, &cuRandStates[tid]);
        sortedRandomValues[tid] = curand_uniform(&cuRandStates[tid]);
        sortedIndices[tid] = startPoint + tid;
    }
}

template<typename DType>
__global__ void shuffle_kernel(DType *shuffle, const uint64_t *shuffledIndices, size_t size, uint64_t startPoint) {
    /*!
     * CUDA Kernel to shuffle the input array.
     *
     * @param shuffle : Pointer to array which is to be shuffled.
     * @param shuffledIndices : Pointer to array of indices for `shuffle` to be shuffled simultaneously.
     * @param size : The size of the `sortedIndices` and `sortedRandomValues` beyond which array is not filled.
     * @param startPoint : Start point, only beyond which `sortedIndices` is filled.
     */
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
    /*!
     * CUDA Kernel to fill an array with random uniform errors.
     *
     * @param fillArray : Pointer to array which is to be shuffled.
     * @param start : Start point, only beyond which `fillArray` is filled.
     * @param seed : The seed value to initialize curand.
     * @param cuRandStates : Pointer to array of cuRandStates objects.
     * @param size : The size of the `fillArray` beyond which array is not filled.
     */
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
    /*!
     * CUDA Kernel to fill an array with indicies (incremental values).
     *
     * @param fillArray : Pointer to array which is to be filled.
     * @param start : Start point, only beyond which `fillArray` is filled.
     * @param size : The size of the `fillArray` beyond which array is not filled.
     */
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        fillArray[tid] = start + tid;
    }
}

#endif//__CUDA_AVAILABLE__
#endif//RLPACK_BINARIES_MEMORY_UTILS_CUDAOFFLOAD_CUH_