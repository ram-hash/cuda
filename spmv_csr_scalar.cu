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


template <typename T>
__global__ void spmv_csr_scalar_kernel(T * d_val, T * d_vector, int * d_cols, int * d_ptr, int N, T * d_out)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = tid; i < N; i += blockDim.x * gridDim.x)
	{
		T t = 0;
		int start = d_ptr[i];
		int end = d_ptr[i + 1];
		// One thread handles all elements of the row assigned to it
		for (int j = start; j < end; j++)
		{
			int col = d_cols[j];
			t += d_val[j] * d_vector[col];
		}
		d_out[i] = t;
	}
}





