#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define CEIL(a,b) ((a-1)/b+1)
#define N 1024
typedef long long int lli;

__global__ void Inclusive_Scan(lli *d_in, lli* d_out)
{
     __shared__ lli sh_array[N];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Copying data from global to shared memory
    sh_array[tid] = d_in[id];
    __syncthreads();

    for(int step = 1; step <= N; step *= 2)
    {
        if(tid >= step)
        {
            lli temp = sh_array[tid-step];
            __syncthreads();
            sh_array[tid] += temp;
        }
        __syncthreads();
    }
    __syncthreads();
    d_in[id] = sh_array[tid];
    if(tid == (N - 1))
        d_out[bid] = d_in[id];
}

// This GPU kernel adds the value d_out[id] to all values in the (id+1)th block of d_in
__global__ void Add(lli *d_in, lli *d_out)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    if(bid > 0)
        d_in[id] += d_out[bid-1];
    __syncthreads();
}

int main()
{
    lli size;
    cout << "Enter size of the array\n";
    cin >> size;
    lli h_in[size],h_out[size];
    int bytes = size * sizeof(lli);
    int reduced_size = (int)ceil(1.0*size/N); 
    int reduced_bytes = reduced_size * sizeof(lli);

    srand(time(0));
    for(lli i=0; i<size; i++)
    {
        h_in[i] = rand()%100;
    }
    for(lli i=0; i<size; i++)
    {
	cout << h_in[i] << " ";
    }
    cout <<"\n";
    
    lli *d_in, *d_out, *d_sum;
    cudaMalloc((void**)&d_in, reduced_size*N*sizeof(lli));  
    cudaMalloc((void**)&d_out, reduced_bytes);
    cudaMalloc((void**)&d_sum, sizeof(lli));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    Inclusive_Scan <<< (int)ceil(1.0*size/1024), 1024>>> (d_in, d_out);

    if(size > N)
    {
        Inclusive_Scan <<< 1, N>>> (d_out, d_sum);
        Add <<< reduced_size, N >>> (d_in, d_out);
    }

    cudaMemcpy(h_out, d_in, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sum);
    cout << "Inclusive Scan Array : \n";
    for(lli i=0; i<size; i++)
        cout << h_out[i] << " ";
 }
    