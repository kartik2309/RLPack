//
// Created by Kartik Rajeshwaran on 2022-11-14.
//

#ifndef RLPACK_BINARIES_UTILS_OPS_ARG_MERGESORT_CUH_
#define RLPACK_BINARIES_UTILS_OPS_ARG_MERGESORT_CUH_

#include <cmath>
#include <omp.h>

template<typename DType>
void arg_mergesort_merge_(DType *arr,
                          uint64_t *index,
                          int64_t left,
                          int64_t mid,
                          int64_t right,
                          int64_t parallelismSizeThreshold) {
  const bool enableParallelism = parallelismSizeThreshold < right;
  const int64_t leftSplitVectorLength = mid - left + 1, rightSplitVectorLength = right - mid;
  float_t leftSplitVector[leftSplitVectorLength], rightSplitVector[rightSplitVectorLength];
  uint64_t leftSplitIndex[leftSplitVectorLength], rightSplitIndex[rightSplitVectorLength];
  {
#pragma omp parallel for if(enableParallelism) default(none) \
firstprivate(arr, index, leftSplitVectorLength, left) \
shared(leftSplitVector, leftSplitIndex)
    for (uint64_t leftVectorIndex = 0; leftVectorIndex < leftSplitVectorLength; leftVectorIndex++) {
      leftSplitVector[leftVectorIndex] = arr[left + leftVectorIndex];
      leftSplitIndex[leftVectorIndex] = index[left + leftVectorIndex];
    }
  }
  {
#pragma omp parallel for if(enableParallelism) default(none) \
firstprivate(arr, index, rightSplitVectorLength, mid) \
shared(rightSplitVector, rightSplitIndex)
    for (uint64_t rightVectorIndex = 0; rightVectorIndex < rightSplitVectorLength; rightVectorIndex++) {
      rightSplitVector[rightVectorIndex] = arr[mid + 1 + rightVectorIndex];
      rightSplitIndex[rightVectorIndex] = index[mid + 1 + rightVectorIndex];
    }
  }
  int64_t leftVectorIndex = 0, rightVectorIndex = 0, mergedVectorIndex = left;
  while (leftVectorIndex < leftSplitVectorLength and rightVectorIndex < rightSplitVectorLength) {
    if (leftSplitVector[leftVectorIndex] <= rightSplitVector[rightVectorIndex]) {
      arr[mergedVectorIndex] = leftSplitVector[leftVectorIndex];
      index[mergedVectorIndex] = leftSplitIndex[leftVectorIndex];
      leftVectorIndex++;
    } else {
      arr[mergedVectorIndex] = rightSplitVector[rightVectorIndex];
      index[mergedVectorIndex] = rightSplitIndex[rightVectorIndex];
      rightVectorIndex++;
    }
    mergedVectorIndex++;
  }
  while (leftVectorIndex < leftSplitVectorLength) {
    arr[mergedVectorIndex] = leftSplitVector[leftVectorIndex];
    index[mergedVectorIndex] = leftSplitIndex[leftVectorIndex];
    leftVectorIndex++;
    mergedVectorIndex++;
  }
  while (rightVectorIndex < rightSplitVectorLength) {
    arr[mergedVectorIndex] = rightSplitVector[rightVectorIndex];
    index[mergedVectorIndex] = rightSplitIndex[rightVectorIndex];
    rightVectorIndex++;
    mergedVectorIndex++;
  }
}

template<typename DType>
void arg_mergesort(DType *arr, uint64_t *index, int64_t begin, int64_t end, int64_t parallelismSizeThreshold) {
  if (begin >= end) {
    return;
  }
  auto mid = begin + (end - begin) / 2;
  bool enableParallelism = parallelismSizeThreshold < end;
#pragma omp parallel sections if(enableParallelism) default(none) \
firstprivate(begin, mid, end, parallelismSizeThreshold) \
shared(arr, index)
  {
#pragma omp section
    {
      arg_mergesort(arr, index, begin, mid, parallelismSizeThreshold);
    }
#pragma omp section
    {
      arg_mergesort(arr, index, mid + 1, end, parallelismSizeThreshold);
    }
  }
  arg_mergesort_merge_(arr, index, begin, mid, end, parallelismSizeThreshold);
}

#endif //RLPACK_BINARIES_UTILS_OPS_ARG_MERGESORT_CUH_
