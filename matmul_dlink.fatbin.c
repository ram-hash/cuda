#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000108,0x0000004801010002,0x00000000000000c0\n"
".quad 0x00000000000000be,0x0000002500010007,0x0000000700000040,0x0000000000002013\n"
".quad 0x0000000000000000,0x0000000000000268,0x00206f2e6e69616d,0x010102464c457fa2\n"
".quad 0x0002660001000733,0xc0230001006600be,0xf500010012000801,0x380040002505250d\n"
".quad 0x0100040040000300,0x72747368732e0000,0x2700082e00626174,0x735f00ff00086d79\n"
".quad 0x766e2e0078646e68,0x2100326f666e692e,0x2e00df004800010f,0x0100402200010003\n"
".quad 0x0108003000322e00,0x722f0400400b1f00,0x0174131113004000,0x000100a82200010e\n"
".quad 0x2a00240600061811,0x0000065700180008,0x0500480f01a80500,0x003801130040a81b\n"
".quad 0x2f0038081500010f, 0x0008801700010006, 0x0000000000000000\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[35];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 2, fatbinData, (void**)__cudaPrelinkedFatbins };
#ifdef __cplusplus
}
#endif
