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
__global__ void spmv_csr_adaptive_kernel(T * d_val, T * d_vector, int * d_cols, int * d_ptr, int N, int * d_rowBlocks, T * d_out)
{
	int startRow = d_rowBlocks[blockIdx.x];
	int nextStartRow = d_rowBlocks[blockIdx.x + 1];
	int num_rows = nextStartRow - startRow;
	int i = threadIdx.x;
	__shared__ volatile T LDS[BlockDim];
	// If the block consists of more than one row then run CSR Stream
	if (num_rows > 1) {
		int nnz = d_ptr[nextStartRow] - d_ptr[startRow];
		int first_col = d_ptr[startRow];

		// Each thread writes to shared memory
		if (i < nnz)
		{
			LDS[i] = d_val[first_col + i] * d_vector[d_cols[first_col + i]];
		}
		__syncthreads();

		// Threads that fall within a range sum up the partial results
		for (int k = startRow + i; k < nextStartRow; k += blockDim.x)
		{
			T temp = 0;
			for (int j = (d_ptr[k] - first_col); j < (d_ptr[k + 1] - first_col); j++) {
				temp = temp + LDS[j];
			}
			d_out[k] = temp;
		}
	}
	// If the block consists of only one row then run CSR Vector
	else {
		// Thread ID in warp
		int rowStart = d_ptr[startRow];
		int rowEnd = d_ptr[nextStartRow];

		T sum = 0;

		// Use all threads in a warp to accumulate multiplied elements
		for (int j = rowStart + i; j < rowEnd; j += BlockDim)
		{
			int col = d_cols[j];
			sum += d_val[j] * d_vector[col];
		}

		LDS[i] = sum;
		__syncthreads();

		// Reduce partial sums
		for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
			__syncthreads();
			if (i < stride)
				LDS[i] += LDS[i + stride];
		}
		// Write result
		if (i == 0)
			d_out[startRow] = LDS[i];
	}
}

int spmv_csr_adaptive_rowblocks(int *ptr, int totalRows, int *rowBlocks)
{
	rowBlocks[0] = 0;
	int sum = 0;
	int last_i = 0;
	int ctr = 1;
	for (int i = 1; i < totalRows; i++) {
		// Count non-zeroes in this row 
		sum += ptr[i] - ptr[i - 1];
		if (sum == BlockDim) {
			// This row fills up LOCAL_SIZE 
			last_i = i;
			rowBlocks[ctr++] = i;
			sum = 0;
		}
		else if (sum > BlockDim) {
			if (i - last_i > 1) {
				// This extra row will not fit 
				rowBlocks[ctr++] = i - 1;
				i--;
			}
			else if (i - last_i == 1)
				// This one row is too large
				rowBlocks[ctr++] = i;
			last_i = i;
			sum = 0;
		}
	}
	rowBlocks[ctr++] = totalRows;
	return ctr;
}