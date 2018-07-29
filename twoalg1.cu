#include "twoalg.h"

void TFISTA(float *t,float *d,float *b,int row,int col,int tube,int rank){     //迭代最小化算法，T为脏数据，d为张量基，b为要学习的系数 T=d*b  t的大小为row*col*tube, d为row*rank*tube, b为rank*col*tube
	
	float *dt = new float[rank*row*tube];    
	for(int i = 0;i<rank*row*tube;i++){
		dt[i] = d[i];
	}                                         //dt作为基的转置
	Ttranspose(d,dt,row,rank,tube);        //转置后的张量为dt
	float *dtxd = new float[rank*rank*tube];  //dtxd用来存放D’×D结果张量
	
	tprod(dt,d,dtxd,rank,rank,row,tube);  //dtxd中存放着D’×D  接下来求它的范数  rank*rank*tube

	//printTensor(rank,rank,tube,dtxd);
	//接下来调用函数求得circ矩阵 大小为 rank*tube × rank*tube
	float *Dc = new float[rank*tube*rank*tube];  //给矩阵分配空间
	
	tensor2maxtr(dtxd,Dc,rank,rank,tube);//传入参数，返回矩阵Dc  rank*tube × rank*tube
	//求Dc的最大奇异值，调用Msvd函数					
	//printTensor(rank*tube,rank*tube,1,Dc);
	float lip = 0.0;    //存放lipschitz常数
	float *S = new float[rank*tube]; //S存放所有奇异值的数组，第一项为最大
	float *U = new float[rank*tube*rank*tube];
	float *V = new float[rank*tube*rank*tube];
	
	Msvd(Dc,U,S,V,rank*tube,rank*tube);	
	
	delete[] U;
	U = nullptr;
	delete[] V;
	V = nullptr;
	
	lip = S[0];    //求出lipschitz常数
	cout<<lip<<endl;  //输出当前的svd看看
	float *dtxt = new float[rank*col*tube];    //存放D'*t
	tprod(dt,t,dtxt,rank,col,row,tube);     //D'*t存放在dtxt中
	
//printTensor(rank,col,tube,dtxt);
	float *c1 = new float[rank*col*tube];
	
	for(int i = 0;i<rank*col*tube;i++){
		c1[i] = b[i];	
	}                           //初始化c1 = b
	float d1 = 1.0;   //初始化t1
	float d2 = 0.0;             
	//float beta = 0.8;
	//float *b1 = new float[rank*col*tube];
 //初始化基本结束开始算法
	float lip1 = 0.0; float eta = 1.01;
	float *dtxdxc = new float[rank*col*tube];

	for(int iter = 1;iter<51;iter++){             //每次学习迭代50次
		lip1 = pow(eta,iter)*lip;
		tprod(dtxd,c1,dtxdxc,rank,col,rank,tube);  //dtxdxc存放D'*D*C;接下来计算梯度
		d2 = (1+sqrt(1+4*d1*d1))/2;
//128 threads   rank*cok*tube/128 blocks
		
		float *d_ddt;
		float *d_c1;
		float *d_dt;
		float *d_b;
		cudaMalloc((void**) &d_ddt,sizeof(float)*rank*col*tube);
		cudaMalloc((void**) &d_c1,sizeof(float)*rank*col*tube);
		cudaMalloc((void**) &d_dt,sizeof(float)*rank*col*tube);
		cudaMalloc((void**) &d_b,sizeof(float)*rank*col*tube);

		cudaMemcpy(d_ddt,dtxdxc,sizeof(float)*rank*col*tube,cudaMemcpyHostToDevice);
		cudaMemcpy(d_c1,c1,sizeof(float)*rank*col*tube,cudaMemcpyHostToDevice);
		cudaMemcpy(d_dt,dtxt,sizeof(float)*rank*col*tube,cudaMemcpyHostToDevice);
		cudaMemcpy(d_b,b,sizeof(float)*rank*col*tube,cudaMemcpyHostToDevice);
		
		dim3 threads(128,1,1);
		dim3 blocks((rank*col*tube+128-1)/128,1,1);

		Tfast<<<blocks,threads>>>(d_ddt,d_c1,d_dt,d_b,lip1,d1,d2,rank*col*tube);
		cudaMemcpy(c1,d_c1,sizeof(float)*rank*col*tube,cudaMemcpyDeviceToHost);
		cudaMemcpy(b,d_b,sizeof(float)*rank*col*tube,cudaMemcpyDeviceToHost);

		d1 = d2;                 //更新d和系数b
		cudaFree(d_ddt);
		cudaFree(d_dt);
		cudaFree(d_c1);
		cudaFree(d_b);
		
	}
		
//printTensor(rank,col,tube,b);

	delete[] dt;	dt = nullptr;
	delete[] dtxd;	dtxd = nullptr;
	delete[] Dc;	Dc = nullptr;
	
	delete[] S;	S = nullptr;
	delete[] c1;	c1 = nullptr;
	//delete[] b1;	b1 = nullptr;

}


void TenDL(cufftComplex *t,float *b,float *s,int m,int n,int k,int r){   //张量基算法，输入脏数据，和学习到的系数张量     t为脏数据的傅里叶变换形式 m×n*k  要学习的基为m*r*k  系数为r*n*k
	//t为脏数据的傅里叶变换，C为学习到的系数参数

	cout<<"进入TenDL"<<endl;
	cufftComplex *b_f = new cufftComplex[m*r*k];    //result  
	cufftComplex *s_f = new cufftComplex[r*n*k];   //s^
	Tfft(s,k,r*n,s_f);    //s->s^
	float *dual_lambda = new float[r];    
	srand(time(NULL));
	for(int i = 0;i<r;i++){
		dual_lambda[i] = 10*fabs(rand()*0.1/(RAND_MAX*0.1));
		//cout<<dual_lambda[i]<<endl;
	}   //random dual_lambda
	float *minx = new float[r];
	cufftComplex *SSt = new cufftComplex[r*r*k];   //s^*s^'
	cufftComplex *XSt = new cufftComplex[m*r*k];   //x^*s^'
	cufftComplex *t_p = new cufftComplex[m*n];	//m*n
	cufftComplex *s_f_p = new cufftComplex[r*n];	//r*n
	cufftComplex *st_f = new cufftComplex[n*r*k];   //transport of s_f  n*r*k
	Tftranspose(s_f,st_f,r,n,k);
	cufftComplex *st_f_p = new cufftComplex[n*r];  // n*r
	cufftComplex *SSt_p = new cufftComplex[r*r];	
	cufftComplex *XSt_p = new cufftComplex[m*r];   
	

	for(int i = 0;i<k;i++){

		for(int j = 0;j<m*n;j++){	
			t_p[j]= t[i*m*n+j]; //xhatk  m*n
		}					
		for(int a = 0;a<r*n;a++){
			s_f_p[a]= s_f[i*r*n+a];  //shatk  r*n
			st_f_p[a]= st_f[i*r*n+a]; //shatk'  n*r
		}
		mul_pro(s_f_p,st_f_p,SSt_p,r, r, n);  //SSt(:,:,i)   r*r	
		for(int v = 0;v<r*r;v++){
			SSt[i*r*r+v] = SSt_p[v];         //collect SSt
		}		
		mul_pro(t_p,st_f_p,XSt_p,m, r, n);  //XSt(:,:,i)   m*r

		for(int c = 0;c<m*r;c++){
			XSt[i*m*r+c] = XSt_p[c];
		}
	}

	//SSt,XSt k,dual_lambda as init value 
        //return x which make obj(lambda) the smallest
	fmincon(minx,dual_lambda,XSt,SSt,m,n,k,r);    // the value is minx

	cufftComplex *SSt_inv = new cufftComplex[r*r];
	cufftComplex *SSt_lam = new cufftComplex[r*r];  //inner variable
	cufftComplex *XSt_p_H = new cufftComplex[r*m];
	cufftComplex *bkt = new cufftComplex[r*m];
	cufftComplex *bkt_H = new cufftComplex[m*r];
	for(int i = 0;i<k;i++){
		
		for(int j = 0;j<r*r;j++){
			SSt_p[j] = SSt[i*r*r+j];	
		}		
		for(int l = 0;l<r;l++){
			for(int v = 0;v<r;v++){
				if(l == v){
					SSt_lam[l*r+v].x = SSt_p[l*r+v].x+minx[v];
					SSt_lam[l*r+v].y = SSt_p[l*r+v].y;
				}else{
					SSt_lam[l*r+v] = SSt_p[l*r+v];			
				}			
			}
		}
		cuinverse(SSt_lam,SSt_inv, r);    //pinv(SStk + Lambda)

		for(int a = 0;a<m*r;a++){
			XSt_p[a] = XSt[i*m*r+a];
		}
		for(int d = 0;d<m;d++){
			for(int e = 0;e<r;e++){
				XSt_p_H[e*m+d].x = XSt_p[d*r+e].x;
				XSt_p_H[e*m+d].y = 0-XSt_p[d*r+e].y;    //XSt_p'		
			}
		}
		mul_pro(SSt_inv,XSt_p_H,bkt,r, m, r);   // Bhatkt = pinv(SStk + Lambda) * XStk'

		for(int b = 0;b<r;b++){
			for(int t = 0;t<m;t++){
				bkt_H[t*r+b].x =bkt[b*m+t].x;
				bkt_H[t*r+b].y = 0-bkt[b*m+t].y;  //bkt_H  m*r	
			}
		}
		for(int v = 0;v<m*r;v++){
			b_f[i*m*r+v] = bkt_H[v];
		}	
	}
	
	//recover b
	Tifft(b, k,m*r,b_f);

	delete[] b_f;		b_f = nullptr;
	delete[] s_f;		s_f = nullptr;
	delete[] dual_lambda;	dual_lambda = nullptr;
	delete[] SSt; 		SSt = nullptr;
	delete[] XSt;		XSt = nullptr;
	delete[] t_p;		t_p = nullptr;
	delete[] s_f_p ;	s_f_p = nullptr;
	delete[] st_f ; 	st_f = nullptr;
	delete[] st_f_p ;	st_f_p = nullptr;
	delete[] SSt_p ;	SSt_p = nullptr;
	delete[] XSt_p ;	XSt_p = nullptr;
	delete[] SSt_lam ;  	SSt_lam = nullptr;
	delete[] SSt_inv ; 	SSt_inv = nullptr;
	delete[] XSt_p_H ;   	XSt_p_H = nullptr;
	delete[] bkt;  		bkt = nullptr;
	delete[] bkt_H ;  	bkt_H = nullptr;
	delete[] minx;		minx = nullptr;


}
