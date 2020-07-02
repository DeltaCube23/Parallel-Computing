#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define CEIL(a,b) ((a-1)/b+1)
#define N 1024

__global__  void Sum(float* d_a,float* d_b,float* d_c,int r,int c)
{
    int x=blockIdx.x*blockDim.x + threadIdx.x;
    int y=blockIdx.y*blockDim.y + threadIdx.y;
    int index=c*y+x;
    if(x<c && y<r)
        d_c[index]=d_a[index]+d_b[index];
}

int main() 
{
    int r,c;
    cout<<"enter row and column : ";
    cin>>r>>c;
    float h_a[r][c], h_b[r][c], h_c[r][c];
    int bytes=r*c*sizeof(float);
    
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
			h_a[i][j]=rand()%1000;
			h_b[i][j]=rand()%1000;
        }
    }
	
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_c, bytes);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(32, 32, 1);
    dim3 grid(CEIL(r, 32), CEIL(c, 32), 1);
    Sum<<<grid, block>>>(d_a,d_b,d_c,r,c);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    bool verify=true;
    for(int i=0;i<r;i++)
    {
	for(int j=0;j<c;j++)
	{
		if(h_c[i][j]!=h_a[i][j]+h_b[i][j])
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

