#include "tprod.h"
#include "fft.h"

void tprod(float *t1,float *t2,float *t,int row, int col, int rank, int tube){      //实现张量乘积，两个为相乘，一个存放结果，row*rank*tube   rank*col*tube    结果为row*col*tube
	
	cufftComplex *t1_f = new cufftComplex[row*rank*tube];
	cufftComplex *t2_f = new cufftComplex[rank*col*tube];      //给两个张量分配空间存放傅里叶变换后的数据
	Tfft(t1,tube,row*rank,t1_f);
	Tfft(t2,tube,rank*col,t2_f); //将两个张量分别作傅里叶变换
	
	cufftComplex *t_f = new cufftComplex[row*col*tube];
	cufftComplex *t1_f_p = new cufftComplex[row*rank];
	cufftComplex *t2_f_p = new cufftComplex[rank*col];
	cufftComplex *t_f_p = new cufftComplex[row*col];  
	for(int i = 0;i<row*col*tube;i++){
		t_f[i].x = 0;
		t_f[i].y = 0;		
	}                               //存放结果的张量进行初始化

	for(int i = 0;i<tube;i++){       //固定第三维数据，实现两个矩阵的乘法
		for(int j = 0;j<row*rank;j++){
			//cufftComplex *t1_f_p = new cufftComplex[row*rank];
			t1_f_p[j]= t1_f[i*row*rank+j];
			
		}					
		for(int k = 0;k<rank*col;k++){
                        //cufftComplex *t2_f_p = new cufftComplex[rank*col];
			t2_f_p[k]= t2_f[i*rank*col+k];
			
		}
		
		//cufftComplex *t_f_p = new cufftComplex[row*col];  
		//for(int w = 0;i<row*col;w++){
		//	t_f_p[w].x = 0;
		//	t_f_p[w].y = 0;		
		//}                                      //存放每一个前面切片的结果,初始化为0
		//接下来调用cublas库来计算矩阵相乘
		
		mul_pro(t1_f_p,t2_f_p,t_f_p,row, col, rank);
		
		for(int v = 0;v<row*col;v++)
			t_f[i*row*col+v] = t_f_p[v]; 	//t_f中为张量积，复数形式
			
				
	}

	delete[] t1_f_p;
	t1_f_p = nullptr;
  	delete[] t2_f_p;	t2_f_p = nullptr;
	delete[] t_f_p;	t_f_p = nullptr;
	delete[] t1_f;	t1_f = nullptr;
  	delete[] t2_f;	t2_f = nullptr;

	Tifft(t,tube,row*col,t_f);           //进行逆傅里叶变换，结果保存在张量t中
	delete[] t_f;	t_f = nullptr;
	

}
