#include "twoalg.h"

void tensor2maxtr(float *T,float *Ac,int a,int b,int c){ 
	for(int i = 0;i<a*c;i++){
		
		for(int j = 0;j<b;j++){

			Ac[i*b*c+j] = T[i*b+j];   //将原矩阵赋值进入Ac
				
		}
	}
	
	for(int k = 1;k<c;k++){
		for(int i = 0;i<a*c;i++){
			for(int j = 0;j<b;j++){
				Ac[i*b*c+k*b+j] = T[((i+(c-k)*a)%(a*c))*b+j];
			}
		}            			//矩阵循环后赋值进入AC中
	}
	cout<<"进入张量转为矩阵了"<<endl;
}



/*void Msvd(float *A,float *U,float *S,float *V,int m,int n){   //实现矩阵的svd，A的大小为m*n
	float *AT = new float[m*n];
	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			AT[j*m+i] = A[i*n+j];
			
		}
	}
	
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;
	gesvdjInfo_t gesvdj_params = NULL;   //创建句柄	
	
	const int lda = m; //矩阵A的主维度
	float *d_A = NULL; // device copy of A 
	float *d_S = NULL; // singular values 
	float *d_U = NULL; // left singular vectors 
	float *d_V = NULL; // right singular vectors 
	int *d_info = NULL; // error info 
	int lwork = 0;
	
	float *d_work = NULL; // devie workspace for gesvdj 
	int info = 0;	// host copy of error info 

	
	const double tol = 1.e-7;
	const int max_sweeps = 15;
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
	const int econ = 0 ;

	
	cusolverDnCreate(&cusolverH);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cusolverH, stream);
	cusolverDnCreateGesvdjInfo(&gesvdj_params);
	cusolverDnXgesvdjSetTolerance(
		gesvdj_params,
		tol);

	cusolverDnXgesvdjSetMaxSweeps(
		gesvdj_params,
		max_sweeps);
	cudaMalloc((void**)&d_A,sizeof(float)*lda*n);
	cudaMalloc((void**)&d_S,sizeof(float)*n);
	cudaMalloc((void**)&d_U,sizeof(float)*lda*m);
	cudaMalloc((void**)&d_V,sizeof(float)*n*n);
	cudaMalloc((void**)&d_info,sizeof(float));

	cudaMemcpy(d_A, AT, sizeof(float)*lda*n,cudaMemcpyHostToDevice); //A传到GPU端
	
	cusolverDnSgesvdj_bufferSize(
		cusolverH,
		jobz, 	// CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only 
			// CUSOLVER_EIG_MODE_VECTOR: compute singular value and singularvectors 
		econ,    // econ = 1 for economy size 
		m,    // nubmer of rows of A, 0 <= m 
		n,   // number of columns of A, 0 <= n 
		d_A,  // m-by-n 
		lda,  // leading dimension of A 
		d_S,  // min(m,n) 
			// the singular values in descending order 
		d_U,   // m-by-m if econ = 0 
			// m-by-min(m,n) if econ = 1 
		lda,    // leading dimension of U, ldu >= max(1,m) 
		d_V,   // n-by-n if econ = 0 
			// n-by-min(m,n) if econ = 1 
		n,   	// leading dimension of V, ldv >= max(1,n) 
		&lwork,
		gesvdj_params);

	cudaMalloc((void**)&d_work , sizeof(float)*lwork);

	cusolverDnSgesvdj(
		cusolverH,
		jobz, // CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only 
			// CUSOLVER_EIG_MODE_VECTOR: compute singular value and singularvectors 
		econ, 	// econ = 1 for economy size 
		m, 	// nubmer of rows of A, 0 <= m 
		n,	// number of columns of A, 0 <= n 
		d_A,	// m-by-n 
		lda,	// leading dimension of A 
		d_S,	// min(m,n) 
			// the singular values in descending order 
		d_U,
			// m-by-m if econ = 0 
			// m-by-min(m,n) if econ = 1 
		lda,
			// leading dimension of U, ldu >= max(1,m) 
		d_V,
			// n-by-n if econ = 0 
			// n-by-min(m,n) if econ = 1 
		n,
			// leading dimension of V, ldv >= max(1,n) 
		d_work,
		lwork,
		d_info,
		gesvdj_params);

	cudaDeviceSynchronize();

	cudaMemcpy(U, d_U, sizeof(float)*lda*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, sizeof(float)*n*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(S, d_S, sizeof(float)*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	if ( 0 == info ){
	printf("gesvdj converges \n");
	}else if ( 0 > info ){
	printf("%d-th parameter is wrong \n", -info);
	exit(1);
	}else{
	printf("WARNING: info = %d : gesvdj does not converge \n", info );
	}
	
	
	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(d_info);
	cudaFree(d_work);
	cout<<"＝＝＝＝＝"<<endl;
	cusolverDnDestroy(cusolverH);
	cout<<"＝＝＝＝＝"<<endl;
	cudaStreamDestroy(stream);
	cout<<"＝＝＝＝＝"<<endl;
	cusolverDnDestroyGesvdjInfo(gesvdj_params);
	cout<<"＝＝＝＝＝"<<endl;
	cudaDeviceReset();
	
	delete[] AT;
	AT = nullptr;
	printf("进入Msvd\n");
}

*/


void Msvd(float *A,float *U,float *S,float *V,int m,int n){   //实现矩阵的svd，A的大小为m*n
	cout<<"＝＝＝＝＝"<<endl;
	cusolverDnHandle_t cusolverH = NULL;   //创建句柄	返回的是Ｖ的共轭转置　按列存储的
	//行数必须大于等于列数
	float *AT = new float[m*n];
	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			AT[j*m+i] = A[i*n+j];
			
		}
	}
	const int lda = m; //矩阵A的主维度
	//显存端分配空间
	
	float *d_A = NULL; /* device copy of A */
	float *d_S = NULL; /* singular values */
	float *d_U = NULL; /* left singular vectors */
	float *d_V = NULL; /* right singular vectors */
	int *devInfo = NULL;
	float *d_work = NULL;
	float *d_rwork = NULL;

	int lwork = 0;
	int info_gpu = 0;
	cusolverDnCreate(&cusolverH);
	
	cudaMalloc((void**)&d_A,sizeof(float)*lda*n);
	cudaMalloc((void**)&d_S,sizeof(float)*n);
	cudaMalloc((void**)&d_U,sizeof(float)*lda*m);
	cudaMalloc((void**)&d_V,sizeof(float)*n*n);
	cudaMalloc((void**)&devInfo,sizeof(int));

	cudaMemcpy(d_A, AT, sizeof(float)*lda*n,cudaMemcpyHostToDevice); //A传到GPU端
	cusolverDnSgesvd_bufferSize(
		cusolverH,
		m,
		n,
		&lwork );
	
	cudaMalloc((void**)&d_work , sizeof(float)*lwork);

	
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	cusolverDnSgesvd (
		cusolverH,
		jobu,
		jobvt,
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		lda, // ldu
		d_V,
		n, // ldvt,
		d_work,
		lwork,
		d_rwork,
		devInfo);

	cudaDeviceSynchronize();

	cudaMemcpy(U, d_U, sizeof(float)*lda*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, sizeof(float)*n*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(S, d_S, sizeof(float)*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	

	//printf("after gesvd: info_gpu = %d\n", info_gpu);

	//printf("=====\n");

	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);

	cusolverDnDestroy(cusolverH);


	cudaDeviceReset();

	delete[] AT;
	AT = nullptr;
	cout<<"Msvd结束"<<endl;

}

void Mfsvd(cufftComplex *A,cufftComplex *U,float *S,cufftComplex *V,int m,int n){   //实现矩阵的svd，A的大小为m*n
	cusolverDnHandle_t cusolverH = NULL;   //创建句柄	返回的是Ｖ的共轭转置　按列存储的
	//行数必须大于等于列数
	cufftComplex *AT = new cufftComplex[m*n];
	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			AT[j*m+i] = A[i*n+j];
			
		}
	}
	const int lda = m; //矩阵A的主维度
	//显存端分配空间
	
	cufftComplex *d_A = NULL; /* device copy of A */
	float *d_S = NULL; /* singular values */
	cufftComplex *d_U = NULL; /* left singular vectors */
	cufftComplex *d_V = NULL; /* right singular vectors */
	int *devInfo = NULL;
	cufftComplex *d_work = NULL;
	float *d_rwork = NULL;

	int lwork = 0;
	int info_gpu = 0;
	cusolverDnCreate(&cusolverH);
	
	cudaMalloc((void**)&d_A,sizeof(cufftComplex)*lda*n);
	cudaMalloc((void**)&d_S,sizeof(float)*n);
	cudaMalloc((void**)&d_U,sizeof(cufftComplex)*lda*m);
	cudaMalloc((void**)&d_V,sizeof(cufftComplex)*n*n);
	cudaMalloc((void**)&devInfo,sizeof(int));

	cudaMemcpy(d_A, AT, sizeof(cufftComplex)*lda*n,cudaMemcpyHostToDevice); //A传到GPU端
	cusolverDnCgesvd_bufferSize(
		cusolverH,
		m,
		n,
		&lwork );
	
	cudaMalloc((void**)&d_work , sizeof(cufftComplex)*lwork);

	
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	cusolverDnCgesvd (
		cusolverH,
		jobu,
		jobvt,
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		lda, // ldu
		d_V,
		n, // ldvt,
		d_work,
		lwork,
		d_rwork,
		devInfo);

	cudaDeviceSynchronize();

	cudaMemcpy(U, d_U, sizeof(cufftComplex)*lda*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, sizeof(cufftComplex)*n*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(S, d_S, sizeof(float)*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	

	//printf("after gesvd: info_gpu = %d\n", info_gpu);

	//printf("=====\n");

	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);
	
	cusolverDnDestroy(cusolverH);
	cudaDeviceReset();
	delete[] AT;
	AT = nullptr;

}

void cuinverse(cufftComplex *A,cufftComplex *A_f,int m){  //A 为原矩阵，A_f为逆矩阵
	cufftComplex *U = new cufftComplex[m*m];  //存放左特征向量
	cufftComplex *V = new cufftComplex[m*m];	//存放右特征向量
	float *S = new float[m];
	cufftComplex *UT = new cufftComplex[m*m];
	cufftComplex *VT = new cufftComplex[m*m];

	Mfsvd(A,U,S,V,m,m);  //实现方矩阵的奇异值分解

	/*for(int i = 0;i<m;i++){
		cout<<S[i]<<" "<<endl;
	}
	cout<<endl;
	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<U[j*m+i].x<<"+"<<U[j*m+i].y<<"i"<<" ";
		}
		cout<<endl;
	}
	cout<<"_____+++____"<<endl;

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<V[j*m+i].x<<"+"<<V[j*m+i].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //V里存的是V的转置
	cout<<"_____"<<endl;
*/
	
	for(int i = 0;i<m;i++){
		if(S[i]>1e-06){
			S[i] = 1/S[i];
		}else{
			S[i] = 0;	
		}
		//cout<<S[i]<<" "<<endl;
	}
	cout<<endl;      //S阵取逆就是取倒数

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			VT[i*m+j].x= V[j*m+i].x;
			VT[i*m+j].y= 0-V[j*m+i].y;
			
		}
	}   
 	/*for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<VT[j*m+i].x<<"+"<<VT[j*m+i].y<<"i"<<" ";
		}
	cout<<endl;
	}
	cout<<" +++"<<endl;
*/
	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			VT[i*m+j].x = (VT[i*m+j].x)*S[i];
			VT[i*m+j].y = (VT[i*m+j].y)*S[i];
		}
	}
	
	/* for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<VT[j*m+i].x<<"+"<<VT[j*m+i].y<<"i"<<" ";
		}
	cout<<endl;
	}
*/
	 for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			UT[i*m+j].x = U[j*m+i].x;  //U的逆就是转置
			UT[i*m+j].y = 0 - U[j*m+i].y;
		}
		
	}

	
	

	/*for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<UT[j*m+i].x<<"+"<<UT[j*m+i].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //U的转置UT
	cout<<"_____"<<endl;
	*/

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			U[j*m+i] = UT[i*m+j];
			
		}
	}          //将UT按行存储

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			V[j*m+i] = VT[i*m+j];
			
		}  //将VT按行存储
	} 

	/*for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<V[i*m+j].x<<"+"<<V[i*m+j].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //V
	cout<<"_____"<<endl;
	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<U[i*m+j].x<<"+"<<U[i*m+j].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //U
	cout<<"_____"<<endl;
*/
	//printfTensor(m,m,1,V);
	//printfTensor(m,m,1,U);
	
	mul_pro(V,U,A_f,m,m,m);//A_f为逆矩阵
	//printfTensor(m,m,1,A_f);
	delete[] U;
	U = nullptr;
	delete[] V;
	V = nullptr;
	delete[] UT;
	UT = nullptr;
	delete[] VT;
	VT = nullptr;
	delete[] S;
	S = nullptr;
	
}


void ceig(float *A,float *V,float *W,int a){   //get eigvalues and vectors

	float *AT = new float[a*a];
	for(int i = 0;i<a;i++){
		for(int j = 0;j<a;j++){
			AT[j*a+i] = A[i*a+j];
		}
	}        //colum store

	float *VT = new float[a*a];
	for(int i = 0;i<a*a;i++){
		VT[i] = 0;
	} 

	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;
	syevjInfo_t syevj_params = NULL;
	
	float *d_A = NULL;   //eigvector
	float *d_W = NULL;  //eigvalue
	int *d_info = NULL; 
	int lwork = 0;
	float *d_work = NULL;
	int info = 0;

	const double tol = 1.e-7;
	const int max_sweeps = 15;
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute
	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	
	
	cusolverDnCreate(&cusolverH);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cusolverH, stream);
	cusolverDnCreateSyevjInfo(&syevj_params);

	cusolverDnXsyevjSetTolerance(
		syevj_params,
		tol);


	/*default value of max. sweeps is 100 */
	cusolverDnXsyevjSetMaxSweeps(
		syevj_params,
		max_sweeps);

	cudaMalloc((void**)&d_A, sizeof(float)*a*a);
	cudaMalloc((void**)&d_W, sizeof(float)*a);
	cudaMalloc((void**)&d_info, sizeof(int));

	cudaMemcpy(d_A, AT, sizeof(float)*a*a,cudaMemcpyHostToDevice);

	cusolverDnSsyevj_bufferSize(
		cusolverH,
		jobz,
		uplo,
		a,
		d_A,
		a,
		d_W,
		&lwork,
		syevj_params);

	cudaMalloc((void**)&d_work, sizeof(float)*lwork);

	cusolverDnSsyevj(
		cusolverH,
		jobz,
		uplo,
		a,
		d_A,
		a,
		d_W,
		d_work,
		lwork,
		d_info,
		syevj_params);
	cudaDeviceSynchronize();

	cudaMemcpy(W, d_W, sizeof(float)*a,cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_A, sizeof(float)*a*a,cudaMemcpyDeviceToHost);
 	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i = 0;i<a;i++){
		for(int j = 0;j<a;j++){
			V[j*a+i] = VT[i*a+j];
		}
	}

	/*if ( 0 == info ){
		printf("syevj converges \n");
	}else if ( 0 > info ){
		printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}else{
		printf("WARNING: info = %d : syevj does not converge \n", info );
	}*/



	if (d_A) cudaFree(d_A);
	if (d_W) cudaFree(d_W);
	if (d_info ) cudaFree(d_info);
	if (d_work ) cudaFree(d_work);

	cusolverDnDestroy(cusolverH);
	cudaStreamDestroy(stream);
	cusolverDnDestroySyevjInfo(syevj_params);
	cudaDeviceReset();

	delete[] AT; AT = nullptr;
	delete[] VT; VT = nullptr;
}

float psnr(float *image1,float *image2,int m,int n,int k){

	float PSNR = 0.0;
	float MSE = 0.0;
	
	float *d_image1,*d_image2;
	cudaMalloc((void**) &d_image1,sizeof(float)*m*n*k);
	cudaMalloc((void**) &d_image2,sizeof(float)*m*n*k);
	cudaMemcpy(d_image1,image1,sizeof(float)*m*n*k,cudaMemcpyHostToDevice);
	cudaMemcpy(d_image2,image2,sizeof(float)*m*n*k,cudaMemcpyHostToDevice);
	dim3 threads(128,1,1);
	dim3 blocks((m*n*k+128-1)/128,1,1);
	/*for(int i = 0;i<m*n*k;i++){
		image1[i] = image1[i]*255;
		image2[i] = image2[i]*255;
	}*/
	PSNR3D<<<blocks,threads>>>(d_image1,d_image2,m*n*k);
	cudaMemcpy(image1,d_image1,sizeof(float)*m*n*k,cudaMemcpyDeviceToHost);
	printTensor(m,n,k,image1);
	for(int j = 0;j<k;j++){
		for(int a = 0;a<m*n;a++){
			//MSE = MSE+((image1[j*m*n+a] - image2[j*m*n+a])*(image1[j*m*n+a] - image2[j*m*n+a]));	
			MSE = MSE+image1[j*m*n+a];	
		}
		MSE = MSE/(m*n);
		PSNR = PSNR+10*log10(255*255/MSE);
		cout<<MSE<<endl;
		cout<<PSNR<<endl;
		MSE = 0.0;

	}
	PSNR = PSNR/k;
	return PSNR;

}

void fmincon(float *minx,float *dual_lambda,cufftComplex *XSt,cufftComplex *SSt,int m,int n,int k,int r){
	//get the best minx+++++++++++++++++++++++++

	//XSt m*r*k  SSt r*r*k
	float *f_real = new float[1];
	float *g_real = new float[r];
	float *H_real = new float[r*r];
	float *step = new float[r];
	// get one time f_real,g_real,H_real  how to updtate dual_lambda??????
	// the follow need to loop
	computelam(f_real,g_real,H_real,dual_lambda,step,XSt,SSt,m,n,k,r);
	

}


void computelam(float *f_real,float *g_real,float *H_real,float *dual_lambda,float *step,cufftComplex *XSt,cufftComplex *SSt,int m,int n,int k,int r){

	cufftComplex *SSt_p = new cufftComplex[r*r];
	cufftComplex *XSt_p = new cufftComplex[m*r];
	cufftComplex *SSt_lam = new cufftComplex[r*r];
	cufftComplex *SSt_inv = new cufftComplex[r*r];
	cufftComplex *XSt_p_H = new cufftComplex[r*r];
	cufftComplex *temp1 = new cufftComplex[m*r];
	cufftComplex *temp2 = new cufftComplex[m*m];
	cufftComplex *f = new cufftComplex[1];
	f[0].x = 0.0; f[0].y = 0.0;
	cufftComplex *temp4 = new cufftComplex[r*r];
	cufftComplex *bkt = new cufftComplex[r*r];
	cufftComplex *bkt_H = new cufftComplex[r*r];
	cufftComplex *g = new cufftComplex[r];
	cufftComplex *H = new cufftComplex[r*r];
	float sum = 0.0;


	// get one time f_real,g_real,H_real  how to updtate dual_lambda??????
	// the follow need to loop

	for(int i = 0;i<k;i++){

		for(int j = 0;j<r*r;j++){
			SSt_p[j] = SSt[i*r*r+j];	
		}		
		for(int l = 0;l<r;l++){
			for(int v = 0;v<r;v++){
				if(l == v){
					//SSt_lam[l*r+v].x = SSt_p[l*r+v].x+dual_lambda[v];
					SSt_lam[l*r+v].x = SSt_p[l*r+v].x+dual_lambda[v];
					SSt_lam[l*r+v].y = SSt_p[l*r+v].y;
				}else{
					SSt_lam[l*r+v] = SSt_p[l*r+v];			
				}			
			}
		}
		cuinverse(SSt_lam,SSt_inv, r);    //pinv(SStk + Lambda)
		//cuinverse(SSt_p,SSt_inv, r);
		//printfTensor(r,r,1,SSt_inv); 
		for(int a = 0;a<m*r;a++){
			XSt_p[a] = XSt[i*m*r+a];
		}
		for(int d = 0;d<m;d++){
			for(int e = 0;e<r;e++){
				XSt_p_H[e*m+d].x = XSt_p[d*r+e].x;
				XSt_p_H[e*m+d].y = 0-XSt_p[d*r+e].y;    //XSt_p'		
			}
		}

		mul_pro(XSt_p,SSt_inv,temp1,m, r, r);  //temp1 = XStk*SSt_inv   m*r
		mul_pro(temp1,XSt_p_H,temp2,m, m, r);	//temp2 = XStk*SSt_inv*XStk'  m*m
		
		for(int h = 0;h<m;h++){
			for(int l = 0;l<m;l++){
				if(h == l){
					f[0].x = f[0].x+temp2[h*m+l].x;
					f[0].y = f[0].y+temp2[h*m+l].y;
				}
			}
		}                  //f =trace(temp2)
		
		mul_pro(SSt_inv,XSt_p_H,bkt,r, m, r);  //bkt r*m
		
		for(int b = 0;b<r;b++){
			for(int t = 0;t<m;t++){
				bkt_H[t*r+b].x = bkt[b*m+t].x;
				bkt_H[t*r+b].y = 0-bkt[b*m+t].y;  //bkt_H  m*r		
			}

		}
		mul_pro(bkt,bkt_H,temp4,r, r, m);   //temp4=bkt*bkt'  r*r
		
		for(int v =0;v<r;v++){
			g[v].x = g[v].x - temp4[v*r+v].x;  //g = g -  diag(Bkt*Bkt');
			g[v].y = g[v].y - temp4[v*r+v].y;
		}
		for(int u = 0;u<r*r;u++){
			H[u].x = H[u].x+2*(temp4[u].x*SSt_inv[u].x - temp4[u].y*SSt_inv[u].y);  
			H[u].y = H[u].y+2*(temp4[u].x*SSt_inv[u].y + temp4[u].y*SSt_inv[u].x);
			H_real[u] = H[u].x;
		}              //H = H +2*(Bkt*Bkt').*(SSt_inv);
	}

	for(int i = 0;i<r;i++){
		g_real[i] = g[i].x+k;
		sum = sum + dual_lambda[i];
	}
	f_real[0] = f[0].x + sum;
	//sum = 0.0; f[0].x = 0.0; f[0].y = 0.0;
        //when get g_real H_real ,a function problem: min{g^Ts + 1/2 s^THs: ||s|| <= delta}
	//we need to get step s from the function 

	//min{g^Ts + 1/2*s^THs: ||s|| <= delta}

	float *alpha = new float[r];
	float *coeff = new float[r];

	for(int i = 0;i<r;i++){
		coeff[i] = 1;				
	}
	float *V = new float[r*r];   //return eigvectors
	float *W = new float[r];    //return eigvalues
	float *FVT = new float[r*r];
	ceig(H,V,W,r);  //V is eigVector,W is vector
	for(int i = 0;i<r;i++){
		for(int j = 0;j<r;j++){
			FVT[j*r+i] = -V[i*r+j]; 
		}
	}

	mulvec_pro(FVT,g_real,alpha,r,1,r);
	
	for(int i = 0;i<r;i++){
		coeff[i] = alpha[i]/W[i];
	}
	mulvec_pro(V,coeff,step,r,1,r);   //we get the step





	delete[] SSt_p ;	SSt_p = nullptr;
	delete[] XSt_p ;	XSt_p = nullptr;
	delete[] SSt_lam ;  	SSt_lam = nullptr;
	delete[] SSt_inv ; 	SSt_inv = nullptr;
	delete[] XSt_p_H ;   	XSt_p_H = nullptr;
	delete[] temp1 ;  	temp1 = nullptr;
	delete[] temp2 ;	temp2 = nullptr;
	delete[] bkt;  		bkt = nullptr;
	delete[] bkt_H ;  	bkt_H = nullptr;
	delete[] temp4 ; 	temp4 = nullptr;
	delete[] f;		f = nullptr;
	delete[] g;		g = nullptr;
	delete[] H;		H = nullptr;
	
}







