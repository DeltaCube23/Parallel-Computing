#include<bits/stdc++.h>
#include<cuda.h>
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)

__global__  void Multiply(int* d_a,int* d_b,int* d_c, int N)
{
    int x=blockIdx.x*blockDim.x + threadIdx.x;
    int y=blockIdx.y*blockDim.y + threadIdx.y;
    int index=x*N+y;
    if(x<N && y<N)
    {
	int res=0;
	for(int i=0;i<N;i++)
	    res += (d_a[x*N+i] * d_b[i*N+y]);
	d_c[index]=res;
    }
}

int main()
{
    int N;
    cout<<"enter size : ";
    cin>>N;
    int h_a[N][N], h_b[N][N], h_c[N][N], h_d[N][N];
    int bytes=N*N*sizeof(int);	
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
	    h_a[i][j]= rand()%10 ;
	    h_b[i][j]= rand()%10 ;
        }
    }
    
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_c, bytes);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(32, 32, 1);
    dim3 grid(CEIL(N, 32), CEIL(N, 32), 1);
    Multiply<<<grid, block>>>(d_a,d_b,d_c,N);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);   
    
    for(int i=0;i<N;i++)
    {
	for(int j=0;j<N;j++)
	{
	    int res=0;
            for(int k=0;k<N;k++)
		res += (h_a[i][k]*h_b[k][j]);
	    h_d[i][j]=res;
	}
    }
	
    bool verify=true;
    for(int i=0;i<N;i++)
    {
	for(int j=0;j<N;j++)
	{
	    if(h_c[i][j]!=h_d[i][j])
	    verify=false;
	}
    }
    
    if(verify)
    cout<<"Result is Correct";
    else
    cout<<"Incorrect Result";
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}