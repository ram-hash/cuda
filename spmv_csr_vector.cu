/*#include <stdio.h>
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
#define BlockDim 1024

template <typename T>
__global__ void spmv_csr_vector_kernel(T * d_val, T * d_vector, int * d_cols, int * d_ptr, int N, T * d_out)
{
	// Thread ID in block
	int t = threadIdx.x;

	// Thread ID in warp
	int lane = t & (warpSize - 1);

	// Number of warps per block
	int warpsPerBlock = blockDim.x / warpSize;

	// One row per warp
	int row = (blockIdx.x * warpsPerBlock) + (t / warpSize);

	__shared__ volatile T vals[BlockDim];

	if (row < N)
	{
		int rowStart = d_ptr[row];
		int rowEnd = d_ptr[row + 1];
		T sum = 0;

		// Use all threads in a warp accumulate multiplied elements
		for (int j = rowStart + lane; j < rowEnd; j += warpSize)
		{
			int col = d_cols[j];
			sum += d_val[j] * d_vector[col];
		}
		vals[t] = sum;
		__syncthreads();

		// Reduce partial sums
		if (lane < 16) vals[t] += vals[t + 16];
		if (lane < 8) vals[t] += vals[t + 8];
		if (lane < 4) vals[t] += vals[t + 4];
		if (lane < 2) vals[t] += vals[t + 2];
		if (lane < 1) vals[t] += vals[t + 1];
		__syncthreads();

		// Write result
		if (lane == 0)
		{
			d_out[row] = vals[t];
		}
	}
}