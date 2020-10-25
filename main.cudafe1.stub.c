#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
static void __device_stub__Z17spmv_pcsr_kernel1IfEvPT_S1_PiiS1_(float *, float *, int *, int, float *);
static void __device_stub__Z17spmv_pcsr_kernel2IfEvPT_PiiS1_(float *, int *, int, float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __device_stub__Z17spmv_pcsr_kernel1IfEvPT_S1_PiiS1_(float *__par0, float *__par1, int *__par2, int __par3, float *__par4){__cudaLaunchPrologue(5);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaLaunch(((char *)((void ( *)(float *, float *, int *, int, float *))spmv_pcsr_kernel1<float> )));}
template<> __specialization_static void __wrapper__device_stub_spmv_pcsr_kernel1<float>( float *&__cuda_0,float *&__cuda_1,int *&__cuda_2,int &__cuda_3,float *&__cuda_4){__device_stub__Z17spmv_pcsr_kernel1IfEvPT_S1_PiiS1_( (float *&)__cuda_0,(float *&)__cuda_1,(int *&)__cuda_2,(int &)__cuda_3,(float *&)__cuda_4);}
static void __device_stub__Z17spmv_pcsr_kernel2IfEvPT_PiiS1_(float *__par0, int *__par1, int __par2, float *__par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(float *, int *, int, float *))spmv_pcsr_kernel2<float> )));}
template<> __specialization_static void __wrapper__device_stub_spmv_pcsr_kernel2<float>( float *&__cuda_0,int *&__cuda_1,int &__cuda_2,float *&__cuda_3){__device_stub__Z17spmv_pcsr_kernel2IfEvPT_PiiS1_( (float *&)__cuda_0,(int *&)__cuda_1,(int &)__cuda_2,(float *&)__cuda_3);}
static void __nv_cudaEntityRegisterCallback(void **__T3){__nv_dummy_param_ref(__T3);__nv_save_fatbinhandle_for_managed_rt(__T3);__cudaRegisterEntry(__T3, ((void ( *)(float *, int *, int, float *))spmv_pcsr_kernel2<float> ), _Z17spmv_pcsr_kernel2IfEvPT_PiiS1_, (-1));__cudaRegisterEntry(__T3, ((void ( *)(float *, float *, int *, int, float *))spmv_pcsr_kernel1<float> ), _Z17spmv_pcsr_kernel1IfEvPT_S1_PiiS1_, (-1));}
static void __sti____cudaRegisterAll(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
