#ifndef GUARD_fft_h
#define GUARD_fft_h
#include <cuda.h>
#include <cufft.h>
#include "head.h"

void Tfft(float *t,int l,int bat,cufftComplex *tf);
void Tifft(float *t,int l,int bat,cufftComplex *tf);
void Ttranspose(float *t,float *temp,int row,int col,int tube);
void printTensor(int m,int n,int k,float *t);
void printfTensor(int m,int n,int k,cufftComplex *t);
void Tftranspose(cufftComplex *tf,cufftComplex *temp,int row,int col,int tube);
void mulvec_pro(float *a,float *b,float *c,int m,int n,int k);
void mul_pro(cufftComplex *a,cufftComplex *b,cufftComplex *c,int row,int col,int rank);
#endif
