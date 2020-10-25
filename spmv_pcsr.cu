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
#define threadsPerBlock 64
#define sizeSharedMemory 8
#define BlockDim 1024
#define ITER 3


template <typename T>
__global__ void spmv_pcsr_kernel1(T * d_val, T * d_vector, int * d_cols, int d_nnz, T * d_v)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int icr = blockDim.x * gridDim.x;
	while (tid < d_nnz) {
		d_v[tid] = d_val[tid] * d_vector[d_cols[tid]];
		tid += icr;
	}
}

template <typename T>
__global__ void spmv_pcsr_kernel2(T * d_v, int * d_ptr, int N, T * d_out)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	int tid = threadIdx.x;
	if(gid>=1024)
	  return;
	
	__shared__ volatile int ptr_s[threadsPerBlock + 1];
	__shared__ volatile T v_s[sizeSharedMemory];

	// Load ptr into the shared memory ptr_s
	ptr_s[tid] = d_ptr[gid];

	// Assign thread 0 of every block to store the pointer for the last row handled by the block into the last shared memory location
	if (tid == 0) {
		if (gid + threadsPerBlock > N) {
			ptr_s[threadsPerBlock] = d_ptr[N];
		}
		else {
			ptr_s[threadsPerBlock] = d_ptr[gid + threadsPerBlock];
		}
	}
	__syncthreads();

	int temp = (ptr_s[threadsPerBlock] - ptr_s[0]) / threadsPerBlock + 1;
	int nlen = min(temp * threadsPerBlock, 1024);
	T sum = 0;
	int maxlen = ptr_s[threadsPerBlock];

	for (int i = ptr_s[0]; i < maxlen; i += nlen) {
		int index = i + tid;
		__syncthreads();
		// Load d_v into the shared memory v_s
		for (int j = 0; j < nlen / threadsPerBlock; j++) {
			if (index < maxlen) {
				v_s[tid + j * threadsPerBlock] = d_v[index];
				index += threadsPerBlock;
			}
		}
		__syncthreads();

		// Sum up the elements for a row
		if (!(ptr_s[tid + 1] <= i || ptr_s[tid] > i + nlen - 1)) {
			int row_s = max(ptr_s[tid] - i, 0);
			int row_e = min(ptr_s[tid + 1] - i, nlen);
			for (int j = row_s; j < row_e; j++) {
				sum += v_s[j];
			}
		}
	}
	// Write result
	d_out[gid] = sum;
}
