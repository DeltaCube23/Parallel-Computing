#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <cmath>
#include <cuda.h>
using namespace std;
#define ll long long int
const int Block_Size = 1024;

// This GPU kernel does blockwise in-place scan 
__global__ void Blelloch_Exclusive_Scan(ll *d_in, ll* d_out)
{
    __shared__ ll sh_array[Block_Size];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Copying data from global to shared memory
    sh_array[tid] = d_in[id];
    __syncthreads();
    /** Performing block-wise in-place Blelloch scan **/
    // First step of Blelloch scan : REDUCTION
    for(int k=2; k <= Block_Size; k *= 2)
    {
        if((tid+1) % k == 0)
        {
            sh_array[tid] = max(sh_array[tid - (k/2)], sh_array[tid]);
        }
        __syncthreads();
    }
    // At the end of reduction, the last element of each block conatins the sum of all elements in that block
    // We store these block-wise sums in d_out
    if(tid == (Block_Size - 1))
    {
        d_out[bid] = sh_array[tid];
        sh_array[tid] = 0;
    }
    __syncthreads();

    // Second step of Blelloch scan : DOWNSWEEP 
    // This is structurally the exact reverse of the reduction step
    for(int k = Block_Size; k >= 2; k /= 2)
    {
        if((tid+1) % k == 0)
        {
            ll temp = sh_array[tid - (k/2)];
            sh_array[tid - (k/2)] = sh_array[tid];
            sh_array[tid] = max(temp, sh_array[tid]);
        }
        __syncthreads();
    }
    // Copying the scan result back into global memory
    d_in[id] = sh_array[tid];
    // d_in now contains blockwise scan result
    __syncthreads();
}

// This GPU kernel adds the value d_out[id] to all values in the (id)th block of d_in
__global__ void Max(ll* d_in, ll* d_out)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    d_in[id] = max(d_out[bid], d_in[id]);
    __syncthreads();
}

int main()
{
    ll *h_in, *h_scan;
    int Size;
    cout << "Enter size of the array.\n";
    cin >> Size;
    //append extra element to cover up for exclusive scan
    Size=Size+1;
    int Reduced_Size = (int)ceil(1.0*Size/Block_Size);
    int Array_Bytes = Size * sizeof(ll);
    int Reduced_Array_Bytes = Reduced_Size * sizeof(ll);
    h_in = (ll*)malloc(Array_Bytes);
    h_scan = (ll*)malloc(Array_Bytes);
    // Populating array with random numbers
    srand(time(0));
    for(ll i=0; i<Size; i++)
    {
        h_in[i] = rand()%100;
    }
    cout << "Input Array : \n";
    for(ll i=0; i<Size; i++)
        cout << h_in[i] << " ";
    cout <<"\n";
    ll *d_in, *d_out, *d_max;

    // GPU Memory allocations
    cudaMalloc((void**)&d_in, Reduced_Size*Block_Size*sizeof(ll));  
    // Padding the input array to the next multiple of Block_Size. 
    // The scan algorithm is not dependent on elements past the end of the array, so we don't have to use a special case for the last block.
    cudaMalloc((void**)&d_out, Reduced_Array_Bytes);
    cudaMalloc((void**)&d_max, sizeof(ll));
    // Copying input array from CPU to GPU
    cudaMemcpy(d_in, h_in, Array_Bytes, cudaMemcpyHostToDevice);
    Blelloch_Exclusive_Scan <<< Reduced_Size, Block_Size >>> (d_in, d_out);

    // After first kernel call, d_in has the blockwise scan results and d_out is an auxiliary array that has the blockwise max
    // Second kernel call is done to scan the blockwise max array 
    // Then the ith value in the resultant scanned blockwise max array is checked with every value in the ith block
    // This addition step is done in the Max() kernel
    // This is required only if size of the array is greater than the block size
    if(Size > Block_Size)
    {
        Blelloch_Exclusive_Scan <<< 1, Block_Size >>> (d_out, d_max);
        Max <<< Reduced_Size, Block_Size >>> (d_in, d_out);
    }

    // Copying the result back to the CPU
    cudaMemcpy(h_scan, d_in, Array_Bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cout << "Exclusive Scan Array : \n";
    for(ll i=1; i<Size; i++)
        cout << h_scan[i] << " ";
    cout <<"\n";
    
}