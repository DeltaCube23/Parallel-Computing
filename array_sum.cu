#include<bits/stdc++.h>
#include<cuda.h>
using namespace std;

#define CEIL(a,b) ((a-1)/b+1)
#define N 1024

__global__ void sum(float* d_a, float* d_b, float* d_c, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index<size)
    d_c[index]=d_a[index]+d_b[index];
}

bool verify(float a[], float b[], float c[], int size)
{
	for(int i=0; i<size; i++)
	{
		if(c[i]!=a[i]+b[i])
		return false;
	}
	return true;
}

int main()
{
    int size;
    cout<<"enter array size : ";
    cin>>size;
    float h_a[size], h_b[size], h_c[size];
    int bytes=size*sizeof(float);
 
    for(int i=0; i<size; i++)
    {
        h_a[i]=rand()%1000;
        h_b[i]=rand()%1000;
	}
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    sum<<<CEIL(size,N), N>>>(d_a, d_b, d_c, size);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    if(verify(h_a, h_b, h_c, size))
    cout<<"Result is Correct";
    else
    cout<<"Incorrect Result";
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}