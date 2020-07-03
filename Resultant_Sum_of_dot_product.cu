#include <iostream>
using namespace std;

__global__ void Dot(int* d_a, int* d_b, int* d_c, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id<size)
	d_c[id]=d_a[id]*d_b[id];
}

__global__ void Add(int* d_c, int* d_out, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;

    __shared__ int a[1024];

    if(id < size)
        a[t_id] = d_c[id];  
  
    __syncthreads();

    for(int s = 512; s>0; s = s/2)
    {
        __syncthreads();
        if(id>=size || id+s>=size)
            continue;
        if(t_id<s)
        {
            a[t_id]+=a[t_id + s];
        }
    }
    __syncthreads();

    if(t_id==0)
    d_out[b_id] = a[t_id];   
}

int main()
{
    int size;
    cout<<"Enter size : ";
    cin>>size;
    int h_a[size], h_b[size], h_ans;
    int bytes=size*sizeof(int);
    int length=(int)ceil(1.0*size/1024);
    for(int i=0;i<size;i++)
    {
        h_a[i]=rand()%10;
	h_b[i]=rand()%10;
    }
    int *d_a, *d_b, *d_c, *d_out, *d_ans;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMalloc((void**)&d_ans, sizeof(int));
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    Dot<<<((int)ceil(1.0*size/1024)), 1024>>>(d_a, d_b, d_c, size);  
    Add<<<((int)ceil(1.0*size/1024)), 1024>>>(d_c, d_out, size);
    Add<<<1, 1024>>>(d_out, d_ans, length);
    cudaMemcpy(&h_ans, d_ans, sizeof(int), cudaMemcpyDeviceToHost); 
	
    int res=0;
    for(int i=0;i<size;i++)
    {
	res+=(h_a[i]*h_b[i]);
    }
    if(h_ans==res)
    cout<<"Correct result";
    else
    cout<<"Invalid";
    cudaFree(d_a);
    cudaFree(d_b); 
    cudaFree(d_c);
    cudaFree(d_out);
    cudaFree(d_ans);
}  