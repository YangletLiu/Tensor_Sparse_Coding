#ifndef GUARD_twoalg_h
#define GUARD_twoalg_h
#include "head.h"
#include <fstream>
#include "fft.h"
#include <cufft.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include "tprod.h"
using namespace std;
void TFISTA(float *t,float *d,float *c,int row,int col,int tube,int rank);
void TenDL(cufftComplex *t,float *b,float *s,int m,int n,int k,int r);
void tensor2maxtr(float *T,float *Ac,int a,int b,int c);
void Msvd(float *A,float *U,float *S,float *V,int m,int n);
void Mfsvd(cufftComplex *A,cufftComplex *U,float *S,cufftComplex *V,int m,int n);
void cuinverse(cufftComplex *A,cufftComplex *A_f,int m);
float psnr(float *image1,float *image2,int m,int n,int k);
extern __global__ void Tfast(float *ddt, float *c1,float *dt,float *b, float lip1, float d1,float d2,float N);
void fmincon(float *minx,float *dual_lambda,cufftComplex *XSt,cufftComplex *SSt,int m,int n,int k,int r);
void computelam(float *f_real,float *g_real,float *H_real,float *dual_lambda,float *step,cufftComplex *XSt,cufftComplex *SSt,int m,int n,int k,int r);
extern __global__ void PSNR3D(float *image1,float *image2,int n);
void ceig(float *A,float *V,float *W,int a);
#endif
