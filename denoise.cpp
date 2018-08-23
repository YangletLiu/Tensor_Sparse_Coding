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
#include "twoalg.h"
#define patsize 5
#define basenum 30  //基的数量

using namespace std;

int main(int argc,char *argv[]){
	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);   //确定张量各个维度的大小
	float *T = new float[a*b*c];
	float *T1 = new float[a*b*c];            //a为25，b为33614，c为5
	float *clean = new float[a*b*c];
	float *clean1 = new float[a*b*c];

	ifstream read(argv[4]); //读入数据文件
	for(int i= 0;i<a*b*c;i++){
		read>>T1[i];   //将数据读到数组中，第四个参数为脏数据X0	
	}

	for(int i = 0;i<c;i++)
		for(int j = 0;j<a;j++)
			for(int k = 0;k<b;k++){
				T[i*a*b+j*b+k] = T1[j*b*c+i*b+k];
			}            //matlab中数据是按照水平切片读的，所以变换为前向切片的，步长为b*c

	delete[] T1;
           
 	ifstream read1(argv[5]); //读入数据文件
	for(int i= 0;i<a*b*c;i++){
		read1>>clean1[i];   //将数据读到数组中，第四个参数为脏数据X0	
	}

	for(int i = 0;i<c;i++)
		for(int j = 0;j<a;j++)
			for(int k = 0;k<b;k++){
				clean[i*a*b+j*b+k] = clean1[j*b*c+i*b+k];
			}            //matlab中数据是按照水平切片读的，所以变换为前向切片的，步长为b*c
	delete[] clean1;

	/*srand((unsigned)time(NULL));
	for(int i = 0;i<a*b*c;i++){
		T[i] = rand()%10;  //随机生成张量中的数据0~10,为Nmsi(加了噪声的脏数据)
	}             //按理说大小为25×N×5,也就是歪的X
	*/
          //初始化张量的基，D，D的大小为25×r×5,r的值设定为30，
	float *T_D = new float[a*basenum*c];    //分配空间
	float *T_D1 = new float[a*basenum*c];

	ifstream read2(argv[6]); //读入数据文件
	for(int i= 0;i<a*basenum*c;i++){
		read2>>T_D1[i];   //将数据读到数组中，为初始化的张量基	
	}

	for(int i = 0;i<c;i++)
		for(int j = 0;j<a;j++)
			for(int k = 0;k<basenum;k++){
				T_D[i*a*basenum+j*basenum+k] = T_D1[j*basenum*c+i*basenum+k];
			}  
	delete[] T_D1;

	/*for(int i = 0;i<a*basenum*c;i++){
		T_D[i] = (rand()%10)/10;	//初始化张量基，每个值都在0~1之间	
	}
	*/
	float *T_C = new float[basenum*b*c];     //初始化张量系数，且为0
	for(int i = 0;i<basenum*b*c;i++){
		T_C[i] = 0;
	}

	
	float *T_a = new float[a*b*c];    //存放回复后的张量
		
	
	cufftComplex *T_a_f= new cufftComplex[a*b*c]; //T_a_f为在cpu上分配的复数类型数据
	Tfft(T,c,a*b,T_a_f); //对T进行fft变换,结果在T_a_f中
	//printTensor(a,b,c,clean);
        //算法开始；
	clock_t start,finish;
	start = clock();
	for(int iter = 1;iter<11;iter++){       //迭代十次，且每次迭代都会调用两个算法TSTA和DL
		//cout<<"iter"<<iter<<endl;
		cout<<"___________________++++__++_+_+_+_+_+______"<<endl;
		TFISTA(T,T_D,T_C, a, b, c, basenum);  //调用第一个算法学习张量系数，传入三个张量
		//printTensor(basenum,b,c,T_C);
		cout<<"我是分割线——————————————————1"<<endl;
		finish = clock();
		cout<<(double)(finish-start)/CLOCKS_PER_SEC<<"s";
		cout<<endl;
		TenDL(T_a_f,T_D,T_C,a,b,c,basenum); //调用第二个算法学习张量基，传入原始fft数据，张量系数，还有个30
		cout<<"我是分割线——————————————————2"<<endl;
		TFISTA(T,T_D,T_C, a, b, c, basenum);  //再调用一次
		cout<<"我是分割线——————————————————3"<<endl;
		tprod(T_D,T_C,T_a, a, b, basenum, c);        //将T_D和T_C乘起来，存放到T_a中,row*rank*tube   								     //rank*col*tube    结果为row*col*tube
		//最后将脏数据和干净的数据作比较 T_a和pure
		cout<<"我是分割线——————————————————4"<<endl;
		cout<<"Iter:"<<iter<<"  current PSNR="<<psnr(T_a,clean,a,b,c)<<endl;
			
	}

	delete[] T;
	T = nullptr;
	delete[] T_D;
	T_D = nullptr;
	delete[] T_C;
	T_C = nullptr;
	delete[] T_a_f;	
	T_a_f = nullptr;
	delete[] clean;
	clean = nullptr;
	delete[] T_a; T_a = nullptr;

}
