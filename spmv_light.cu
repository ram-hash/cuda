/*
#include <stdio.h>
#include <math.h>
#include <time.h> 
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "book.h"
#include "cusparse.h" 
*/
template <typename T, int THREADS_PER_VECTOR, int MAX_NUM_VECTORS_PER_BLOCK>
__global__ void spmv_light_kernel(int* cudaRowCounter, int* d_ptr, int* d_cols, T* d_val, T* d_vector, T* d_out, int N) {
	int i;
	T sum;
	int row;
	int rowStart, rowEnd;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & 31;	//lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR;	//vector index in the warp

	__shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

	// Get the row index
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	// Broadcast the value to other threads in the same warp and compute the row index of each vector
	row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

	while (row < N) {

		// Use two threads to fetch the row offset
		if (laneId < 2) {
			space[vectorId][laneId] = d_ptr[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		sum = 0;
		// Compute dot product
		if (THREADS_PER_VECTOR == 32) {

			// Ensure aligned memory access
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			// Process the unaligned part
			if (i >= rowStart && i < rowEnd) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}

			// Process the aligned part
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		}
		else {
			for (i = rowStart + laneId; i < rowEnd; i +=
				THREADS_PER_VECTOR) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		}
		// Intra-vector reduction
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down_sync(0xffffffff, sum, i);
		}

		// Save the results
		if (laneId == 0) {
			d_out[row] = sum;
		}

		// Get a new row index
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		// Broadcast the row index to the other threads in the same warp and compute the row index of each vector
		row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

	}
}
