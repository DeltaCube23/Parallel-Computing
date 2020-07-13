#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

typedef long long int lli;

__global__ void Inclusive_Scan(lli *d_in, lli* d_out, lli size, lli i)
{
    lli id = blockIdx.x * blockDim.x + threadIdx.x;
    lli step = 1<<i;
    if(id<size)
    {
        if(id >= step)
        {
            if(d_in[id]<d_in[id-step])
		d_out[id]=d_in[id-step];
	    else
		d_out[id]=d_in[id];
        }
        else
        {
            d_out[id]=d_in[id];
        }
    }
    __syncthreads();
}

int main()
{
    lli size;
    cout << "Enter size of the array\n";
    cin >> size;
    lli h_in[size],h_out[size];
    lli bytes = size * sizeof(lli);
    
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
    
    lli *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    lli iterations = (lli)floor(log2((double)size)) + 1;
    for(lli i=0; i<iterations; i++)
    {
        Inclusive_Scan <<< (int)ceil(1.0*size/1024), 1024>>> (d_in, d_out, size, i);
        cudaMemcpy(d_in, d_out, bytes, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cout << "Inclusive Scan Array : \n";
    for(lli i=0; i<size; i++)
        cout << h_out[i] << " ";
 }
    