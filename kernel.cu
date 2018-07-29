#include "twoalg.h"

__global__ void Tfast(float *ddt, float *c1,float *dt,float *b, float lip1, float d1,float d2,float N){
	float beta = 0.8;
	int i = blockIdx.x*blockDim.x+threadIdx.x;	
	if(i<N){	
		ddt[i] = ddt[i]-dt[i];   //gradient
		ddt[i] = c1[i]-(ddt[i]/lip1);  //temp
		if(ddt[i]>0){
			dt[i] = 1;
		}else if(ddt[i] == 0){
			dt[i] = 0;       //sign
		}else{
			dt[i]= -1;
		}
		if((fabs(ddt[i]))-(beta/lip1)>0){
			ddt[i] = fabs(ddt[i])-(beta/lip1);
		}else{
			ddt[i] = 0;   //max
		}
		ddt[i] = dt[i]*ddt[i];  //b1
		c1[i] = ddt[i]+((d1-1)/d2)*(ddt[i]-b[i]);
		b[i] = ddt[i];  //return c1  b
	}

}

__global__ void PSNR3D(float *image1,float *image2,int n){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		image1[i] = (image1[i]*255-image2[i]*255)*(image1[i]*255-image2[i]*255);
	}
}
