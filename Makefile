 This Makefile assumes the following module files are loaded:
#
# GCC
# CUDA
#
# This Makefile will only work if executed on a GPU node.
#

NVCC = nvcc

NVCCFLAGS = -O0 -g -G -keep

LFLAGS = -lm -lcusparse -keep

# Compiler-specific flags (by default, we always use sm_37)
GENCODE_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GENCODE = $(GENCODE_SM37)

.SUFFIXES : .cu .ptx

BINARIES = matmul

matmul: main.o spmv_csr_adapative.o spmv_csr_scalar.o spmv_csr_vector.o spmv_light.o spmv_pcsr.o spmv.cuh
	$(NVCC) $(GENCODE) $(LFLAGS) $(NVCCFLAGS) -o $@ $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean: 
	rm -f *.o $(BINARIES) 

run: matmul
	./$(BINARIES) csr.txt values.txt col_ind.txt row_ptr.txt true_y.txt x.txt


