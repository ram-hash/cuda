/*

sparse_matrix.cu:
	Cuda implementation Sparse Matrix Multiplication by Vector

compile & run:
	nvcc sparse_matrix.cu -o sparse_matrix.sh -lm && ./sparse_matrix.sh 32768 256 256 1


input:
	NNZ: None Zero Values
	ROWS: The number of Rows (max 1024)
	COLS: The number of Columns (max 1024)
	DEBUG: 1 to debug, 0 to no-debug

output:
	Time in MS
	Throughput in GFLOPS

author:     Ivan Reyes-Amezcua
date:       June, 2020

*/

#include <stdio.h>
#include <math.h>
#include <time.h> 
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h" 
#include"spmv_csr_scalar.cu"
#include"spmv_pcsr.cu"



//#define  NNZ  65000	// Non Zero Values
//#define  NUM_ROWS  512	// rows
//#define  NUM_COLS  512
#define BlockDim 1024
using namespace std;
void cusparseSPMV(int NUM_ROWS,int NUM_COLS,int NNZ,float *values,int *col_index,int *row_ptrs,float *x,float *y,float *hy) {
	cusparseHandle_t     handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void*                dBuffer = NULL;
	size_t               bufferSize = 0;
	float     alpha = 1.0f;
	float     beta = 0.0f;
	cusparseCreate(&handle);
	cusparseCreateCsr(&matA, NUM_ROWS, NUM_COLS, NNZ,
		row_ptrs, col_index, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
	cusparseCreateDnVec(&vecX, NUM_COLS, x, CUDA_R_32F);
	cusparseCreateDnVec(&vecY, NUM_ROWS, y, CUDA_R_32F);
	cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
	cudaMalloc(&dBuffer, bufferSize);
	cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_MV_ALG_DEFAULT, dBuffer);
	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cudaMemcpy(hy, y, NUM_ROWS * sizeof(float), cudaMemcpyDeviceToHost);

}
//template <typename T>
/*__global__ void spmv_csr_adaptive_kernel(T * d_val, T * d_vector, int * d_cols, int * d_ptr, int N, int * d_rowBlocks, T * d_out)
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
*/
int print_vector(int *x, int number) {
	for (int i = 0; i < number; i++) {
		cout << x[i] << endl;
	}
	return 0;
}
float print_vector(float *x, int number) {
	for (int i = 0; i < number; i++) {
		cout << x[i] << endl;
	}
	return 0;
}
float print_error(float *true_y, float *y,int NUM_ROWS,float density ) {
	int errors = 0;   // count of errors
	float e = 0.1;  // tolerance to error
	for (int i = 0; i < NUM_ROWS; i++) {
		if (abs(true_y[i] - y[i]) > e ){
			errors++;
			//if(debug == 1)
			printf("Error in Y%d, True: %f, Calc: %f\n", i, true_y[i], y[i]);
		}
		else if (i < 10) {
			printf("Y%d, True: %f, Calc: %f\n", i, true_y[i], y[i]);
		}
	}

	float error_rate = ((float)errors / (float)NUM_ROWS) * 100.0;
	//float density = ((float)NNZ/((float)NUM_COLS*(float)NUM_ROWS))*100.0;
	printf("\nM. Density: %0.2f%%, #Ys: %d, Errors: %d, Error Rate: %0.2f%%\n", density, NUM_ROWS, errors, error_rate);
	return error_rate;

}



/*__global__ void spmv(int num_rows, int num_cols, int *row_ptrs,
	int *col_index, float *values, float *x, float *y) {
	extern __shared__ float ss_sum[]; 
	int tid = threadIdx.x;  											// Local: Thread ID
	int g_tid = threadIdx.x + row_ptrs[blockIdx.x];  					// Global: Thread ID + offset in row
	int NNZ_in_row = row_ptrs[blockIdx.x + 1] - row_ptrs[blockIdx.x];	// Non-zero values in current row-block
	 ss_sum[tid] = 0.0;  
	__syncthreads();

	// TODO: check col_index vector, possible memory issue
	if (tid < NNZ_in_row)
		ss_sum[tid] = values[g_tid] * x[col_index[g_tid]]; // Map: value[n] * X[index[n]]

	__syncthreads();
	
	// Inclusive Scan
	float temp=0;
	for (int j = 1; j < blockDim.x; j *= 2 ){
		if ( (tid - j) >= 0)
			temp = ss_sum[tid - j];
		__syncthreads();
	    if ( (tid - j) >= 0)
			ss_sum[tid] += temp;
		__syncthreads();
	}

	// Save the result of Row-Block on global memory
    if(tid == blockDim.x - 1)
		y[blockIdx.x] = ss_sum[tid];
}
*/
int main(int argc, char *argv[]) {

	// Get and validate arguments
			// 1 for debug, 0 for NO-debug
	if(argc<7)
	{
	 cout<<"input parameters: csr_file values_file col_index_file row_ptr_file true_y_file x_file"<<endl;
	 //exit(0);
	}
	float density = 0;
	int  NUM_ROWS = 0;
	int  NUM_COLS = 0;
	int  NNZ = 0;
	std::ifstream csrfile(argv[1]);
	csrfile >> NUM_ROWS >> NUM_COLS >> density >>NNZ;
	cout << NUM_ROWS <<" "<< NUM_COLS << " "<<density <<" " <<NNZ << endl;
	 float* values= (float*)malloc(NNZ * sizeof(float));  				// CSR format
	 int *col_index=(int*)malloc(NNZ * sizeof(int)); 				// CSR format
	 int *row_ptrs=(int*)malloc(NNZ* sizeof(int));  		// CSR format
	//float *x= (float*)malloc((NUM_COLS + 1) * sizeof(float)); 
	float *x = (float*)malloc((NUM_ROWS)* sizeof(float));// the vector to multiply
	float *y= (float*)malloc((NUM_ROWS) * sizeof(float));  				// the output
	float *true_y= (float*)malloc((NUM_ROWS) * sizeof(float));  
	float *v= (float*)malloc((NUM_ROWS) * sizeof(float));// the true Y results of operation
	//for (int i = 0; i < NNZ; i++) {
		//float aux1, aux2;
		//csrfile >> values[i] >> aux1 >> aux2;
		//col_index[i] = (int)aux1;
		//row_ptrs[i] = (int)aux2;
	//}
	// Declare GPU memory pointers
	float *d_values;
	float *d_x;
	float *d_y;
	float *d_v;
	int *d_col_index;
	int *d_row_ptrs;

	// Allocate GPU memory
	int r1 = cudaMalloc((void **) &d_values, NNZ*sizeof( float ));
	int r2 = cudaMalloc((void **) &d_x, NUM_COLS *sizeof( float ));
	int r3 = cudaMalloc((void **) &d_y, 1024*sizeof( float ));
	int r4 = cudaMalloc((void **) &d_col_index, NNZ*sizeof( int ));
	int r5 = cudaMalloc((void **) &d_row_ptrs, (NUM_ROWS + 1)*sizeof( int ));
	int r6= cudaMalloc((void **)&d_v, NUM_ROWS  * sizeof(int));
	//if( r1 || r2 || r3 || r4 || r5 ||r6) {
	/*if(r1==cudaErrorNoDevice ){
	   printf("%d\n",r1);
	   exit( 0 );
	}
	if(r2){
		printf( "Error2 allocating memory in GPU\n" );
		exit( 0 );
	 }
	 if(r3){
		printf( "Error3 allocating memory in GPU\n" );
		exit( 0 );
	 }
	 if(r4){
		printf( "Error4 allocating memory in GPU\n" );
		exit( 0 );
	 }
	 if(r5){
		printf( "Error5 allocating memory in GPU\n" );
		exit( 0 );
	 }
	 if(r6){
		printf( "Error6 allocating memory in GPU\n" );
		exit( 0 );
	 }
	*/

	// Read the Values and Index:
	
	std::ifstream values_file(argv[2]);
	std::ifstream col_ind_file(argv[3]);
	//std::ifstream coo_file("d:/coo.txt");
    for (int i = 0; i < NNZ; i++) {
		values_file >> values[i];

		float aux;
		col_ind_file >> aux;
		col_index[i] = (int) aux;
	}
	cout << "values:" << endl;
	print_vector(values,NNZ);
    cout << "col_index:" << endl;
	print_vector(col_index, NNZ);
	//for(int i=0;i<NNZ;i++)
	// Read the row_ptr and the True Ys:
	std::ifstream row_ptr_file(argv[4]);
	std::ifstream true_y_file(argv[5]);
    for (int i = 0; i < (NUM_ROWS + 1); i++) {
		float aux;

		row_ptr_file >> aux;
		true_y_file >> true_y[i];

		row_ptrs[i] = (int) aux;
		//true_y[i] = (int) aux2;
	}
	//cout << "true_y:" << endl;
	//print_vector(true_y, NUM_ROWS);

	

	// Read the X values:
	std::ifstream x_file(argv[6]);
	for (int i = 0; i < NUM_ROWS; i++) {
		x_file >> x[i];
	}
	cout << "x:" << endl;
	print_vector(x, NUM_ROWS);
	// Transfer the arrays to the GPU:
	cudaMemcpy(d_values, values, NNZ*sizeof( float ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, NUM_COLS *sizeof( float ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, NUM_ROWS *sizeof( float ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_index, col_index, NNZ*sizeof( int ), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_ptrs, row_ptrs, (NUM_ROWS + 1)*sizeof( int ), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_v, v, NNZ * sizeof(float), cudaMemcpyHostToDevice);

	// Start Time:
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//cusparseSPMV(NUM_ROWS, NUM_COLS, NNZ, d_values, d_col_index, d_row_ptrs, d_x, d_y,true_y);
	//cout << "true_y:"<< endl;
	//print_vector(true_y, NUM_ROWS);


	// Call to kernel:
	int size_sharedmem = NUM_COLS *sizeof(int); 
	float size_sharedmem1 = NNZ * sizeof(float);// Size of shared memory
	float size_sharedmem2 = 1024* sizeof(float);// Size of shared memory
	//spmv<<<NUM_ROWS/2, NUM_COLS/2, size_sharedmem>>>(NUM_ROWS, NUM_COLS, d_row_ptrs, d_col_index, d_values, d_x, d_y);
	//spmv_csr_kernel << <NUM_ROWS, NUM_COLS, size_sharedmem1 >> > (d_values, d_x, d_col_index, d_row_ptrs, NNZ, d_y);
	spmv_pcsr_kernel1 << < NUM_ROWS, NUM_COLS >> > (d_values, d_x, d_col_index, NNZ, d_v);
	cout << "v:" << endl;
	cudaMemcpy(v, d_v, NUM_ROWS * sizeof(float), cudaMemcpyDeviceToHost);
	print_vector(v, NUM_ROWS);
	spmv_pcsr_kernel2 << < NUM_ROWS, NUM_COLS , size_sharedmem2>> > (d_v, d_row_ptrs,NNZ,d_y);
	//spmv_csr_scalar_kernel << < NUM_ROWS, NUM_COLS >> > (d_values, d_x, d_col_index, d_row_ptrs, NNZ, d_y);
	cudaMemcpy(y, d_y, NUM_ROWS* sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//cout << "y:" << endl;
	//print_vector(y, NUM_ROWS);


	// Stop Time:
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Transfer the values to the CPU:
	//cudaMemcpy(y, d_y, NUM_ROWS * sizeof(float), cudaMemcpyDeviceToHost);

	// Get the error:
	/*int errors = 0;   // count of errors
	float e = 500.0;  // tolerance to error
	for (int i = 0; i < NUM_ROWS; i++) {
		if (abs(true_y[i] - y[i]) > e) {
			errors++;
			//if(debug == 1)
				print_vector("Error in Y%d, True: %f, Calc: %f\n", i, true_y[i], y[i]);
		} else if ( i < 10) {
			print_vector("Y%d, True: %f, Calc: %f\n", i, true_y[i], y[i]);
		}
	}
	
	float error_rate = ((float)errors/(float)NUM_ROWS) * 100.0;
	//float density = ((float)NNZ/((float)NUM_COLS*(float)NUM_ROWS))*100.0;
	print_vector("\nM. Density: %0.2f%%, #Ys: %d, Errors: %d, Error Rate: %0.2f%%\n", density, NUM_ROWS, errors, error_rate);
	*/
	float error_rate = 0;
	error_rate=print_error(true_y, y, NUM_ROWS,density);
	// Free Memory
	cudaFree( d_values );
	cudaFree( d_x );
	cudaFree( d_y );
	cudaFree( d_col_index );
	cudaFree( d_row_ptrs );
	cudaFree(d_v);

	// Calculate Throughput:
	float bw;
	bw = (float )NUM_ROWS*(float )NUM_COLS*log2((float)NUM_COLS);
	bw /= milliseconds * 1000000.0;
	printf( "\nSpmV GPU execution time: %7.3f ms, Throughput: %6.2f GFLOPS\n\n", milliseconds, bw ); 

	// Store Runtime
	FILE *pFile = fopen("GPU_results.txt","a");
    fprintf(pFile, "%d, %0.2f, %0.2f, %d, %d, %7.3f, %6.2f\n", NNZ, density, error_rate, NUM_COLS, NUM_ROWS, milliseconds, bw);
	fclose(pFile);
	free(values);
	free(col_index);
	free(row_ptrs);
	free(x);
	free(y);
	free(true_y);
    return 0;
}