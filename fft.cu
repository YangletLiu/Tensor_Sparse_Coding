#include "fft.h"


void Tfft(float *t,int l,int bat,cufftComplex *tf){   //t为数据，l为tube，a*b为个数，tf为结果
	cufftComplex *t_f = new cufftComplex[l*bat];  //中间变量，存放转换后的复数数据
	for(int i = 0;i<bat;i++)
		for(int j = 0;j<l;j++){
			t_f[i*l+j].x = t[j*bat+i];  //t为按行存储的
			t_f[i*l+j].y = 0;   //c2c，张量变为复数显示且进行数据的转换，变为mode3的tube
	}
	cufftComplex *d_f;
	cudaMalloc((void**)&d_f,l*bat*sizeof(cufftComplex));  //显存上分配空间
	cudaMemcpy(d_f,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice); //要变换的数据传到显存上
	cufftHandle plan = 0; //创建句柄
	cufftPlan1d(&plan,l,CUFFT_C2C,bat);
	cufftExecC2C(plan,(cufftComplex *)d_f,(cufftComplex *)d_f,CUFFT_FORWARD); //执行
	cudaDeviceSynchronize();
	cudaMemcpy(t_f,d_f,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost); //传回数据
	cufftDestroy(plan);//删除上下文
	cudaFree(d_f);
    //目前t_f中存放傅里叶变换后的张量数据，下一步将其转换回原来的形式。
	for(int i =0;i<bat;i++)
		for(int j = 0;j<l;j++){
			tf[j*bat+i] = t_f[i*l+j];			
		}
	delete[] t_f; //动态释放t_f
	t_f = nullptr;
	
	
}

void Tifft(float *t,int l,int bat,cufftComplex *tf){   //t存放结果，tf存放傅里叶变换的数据来转换
	cufftComplex *t_f = new cufftComplex[l*bat];   //中间变量，存放转换后的复数数据来进行变换
	for(int i =0;i<bat;i++)
		for(int j = 0;j<l;j++){
			t_f[i*l+j]= tf[j*bat+i];  //变成mode3的形式
			
		}
		
	cufftComplex *d_f;
	cudaMalloc((void **)&d_f,sizeof(cufftComplex)*l*bat);  //显存上分配空间
	cudaMemcpy(d_f,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice);  //给显存空间传入数据
	cufftHandle plan = 1; //句柄
	cufftPlan1d(&plan,l,CUFFT_C2C,bat);
	cufftExecC2C(plan,(cufftComplex *)d_f,(cufftComplex *)d_f,CUFFT_INVERSE);
	cudaDeviceSynchronize();
	cudaMemcpy(t_f,d_f,sizeof(cufftComplex)*bat*l,cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(d_f);
	//将结果为复数形式的数据转换到t中
	for(int i = 0;i<bat;i++)
		for(int j = 0;j<l;j++){
			t[j*bat+i] = t_f[i*l+j].x/l;
			
		}
	delete[] t_f;    //释放
	t_f = nullptr;

}


void Ttranspose(float *t,float *temp,int row,int col,int tube){      //实现张量的转置操作，输入张量的各种的信息
	/*for(int i= 0;i<row*col*tube;i++){
		temp[i] = t1[i];	
	}*/
	
	for(int i = 0;i<row;i++)
		for(int j = 0;j<col;j++){
			temp[j*row+i] = t[i*col+j];   //等号后面为原张量切片，第一个切片直接转置
	}
	for(int k = 1;k<tube;k++)
		for(int i = 0;i<row;i++)
			for(int j = 0;j<col;j++){
				temp[k*col*row+j*row+i] = t[(tube-k)*col*row+i*col+j] ;      //
			}
	//cout<<"进入张量转置了"<<endl;

}
void mul_pro(cufftComplex *a,cufftComplex *b,cufftComplex *c,int row,int col,int rank){
      //进行普通矩阵乘，调用cublas库来实现，啊a*b=c  a为row*rank b为rank*col c为row*col
	//注意输入应该为b*a，这样结果传到CPU端才是按行存储的c
	
	cufftComplex *alpha = new cufftComplex[1];
	for(int i = 0;i<1;i++){
		alpha[i].x=1; alpha[i].y=0;
	}	

	 
   	cufftComplex *beta = new cufftComplex[1];
	for(int i = 0;i<1;i++){
		beta[i].x=0; beta[i].y=0;
	} 
	
	 cufftComplex *d_a;
	 cufftComplex *d_b;
	 cufftComplex *d_c;

	cudaMalloc((void**)&d_a,row*rank*sizeof(cufftComplex));  
    	cudaMalloc((void**)&d_b,rank*col*sizeof(cufftComplex));  
    	cudaMalloc((void**)&d_c,row*col*sizeof(cufftComplex));     //显存分配空间
    	cudaMemcpy(d_a,a,row*rank*sizeof(cufftComplex),cudaMemcpyHostToDevice);  
    	cudaMemcpy(d_b,b,rank*col*sizeof(cufftComplex),cudaMemcpyHostToDevice);      //数据传到显存
    	
	cublasHandle_t handle;                  
    	cublasCreate(&handle);  
	cublasCgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		col,
		row,
		rank,
		alpha,
		d_b,
		col,
		d_a,
		rank,
		beta,
		d_c,
		col);  
   	cudaMemcpy(c,d_c,row*col*sizeof(cufftComplex),cudaMemcpyDeviceToHost);    //数据传回
	
	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);             //释放空间
	delete[] alpha;
	alpha = nullptr;
	delete[] beta;
	beta = nullptr;
	

}

void mulvec_pro(float *a,float *b,float *c,int m,int n,int k){
                       //m*k k*n m*n          row  col  rank 
	float alpha = 1; 
	float beta = 0;
	
	float *d_a;
	float *d_b;
	float *d_c;

	cudaMalloc((void**)&d_a,m*k*sizeof(float));  
    	cudaMalloc((void**)&d_b,k*n*sizeof(float));  
    	cudaMalloc((void**)&d_c,m*n*sizeof(float));     //显存分配空间
    	cudaMemcpy(d_a,a,m*k*sizeof(float),cudaMemcpyHostToDevice);  
    	cudaMemcpy(d_b,b,k*n*sizeof(float),cudaMemcpyHostToDevice);      //数据传到显存
	cublasHandle_t handle;                  
    	cublasCreate(&handle);  
    	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		n,
		m,
		k,
		&alpha,
		d_b,
		n,
		d_a,
		k,
		&beta,
		d_c,
		n);  
   	cudaMemcpy(c,d_c,m*n*sizeof(float),cudaMemcpyDeviceToHost);    //数据传回
	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);             //释放空间


}



void printTensor(int m,int n,int k,float *t){
	for(int i = 0;i<k;i++){
		for(int j = 0;j<m;j++){
			for(int l = 0;l<n;l++){
				cout<<t[i*m*n+j*n+l]<<" ";
			}
			cout<<endl;
			cout<<endl;
		}
	cout<<"______________________________________________________________________"<<endl;
	}

}

void printfTensor(int m,int n,int k,cufftComplex *t){
	for(int i = 0;i<k;i++){
		for(int j = 0;j<m;j++){
			for(int l = 0;l<n;l++){
				cout<<t[i*m*n+j*n+l].x<<"+"<<t[i*m*n+j*n+l].y<<"i"<<"  ";
			}
			cout<<endl;
			cout<<endl;
		}
	cout<<"_____________________________________________________________________"<<endl;
	}

}

void Tftranspose(cufftComplex *tf,cufftComplex *temp,int row,int col,int tube){    
	
//复数张量共轭转置,temp存放结果
	for(int i = 0;i<tube;i++)
		for(int j = 0;j<row;j++)
			for(int k = 0;k<col;k++){
				temp[i*col*row+k*row+j].x = tf[i*col*row+j*col+k].x;
				temp[i*col*row+k*row+j].y = 0 - tf[i*col*row+j*col+k].y;
			} 	
			
}




