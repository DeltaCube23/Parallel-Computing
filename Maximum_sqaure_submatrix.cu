%%cu
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)
#define THREAD 1024

// Computes column wise sequence of continious 1's  Eg :- [1 0 1 1 1 0] column will become [1 0 1 2 3 0]
__global__ void downsweep(int *d_b, int m, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n)
    {
		for(int i = 1; i < m; i++)
		{
			if(d_b[i*n + index] == 0)
			continue;
			else
			d_b[i*n + index] = d_b[(i-1)*n + index] + 1;
		}
	}
}

// Computes row wise sequence of continious 1's  Eg :- [1 1 0 1 0 1 1 1] column will become [1 2 0 1 0 1 2 3]
__global__ void sidesweep(int *d_c, int m, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < m)
    {
		for(int i = 1; i < n; i++)
		{
			if(d_c[index*n + i] == 0)
			continue;
			else
			d_c[index*n + i] = d_c[index*n + i - 1] + 1;
		}
	}
}

// computes 1 row at a time using syncthreads
// value[i][j] = min(value[i-1][j-1], value[i-1][j], value[i][j-1]) + 1
// value[i-1][j] is computed during column downsweep
// value[i][j-1] is computed during row sidesweep
// value[i-1][j-1] is computed in previous iteration and 
// since we are syncronizing after each iteration there won't be errors
__global__ void maxsquare(int *d_d, int *d_c, int *d_b, int *d_max, int m, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	d_max[index] = 0;
	if(index < n && index > 0)
	{
		for(int i = 1; i < m; i++)
		{
			if(d_b[i*n + index] != 0)
			{
				//to find the minimum of the 3 neighbours
				int temp = d_b[i*n + index]-1;
				if(d_c[i*n + index] - 1 < temp)
					temp = d_c[i*n + index]-1;
				if(d_d[(i-1)*n + index - 1] < temp)
					temp = d_d[(i-1)*n + index - 1];
				d_d[i*n + index] = temp+1;
				
				if(d_d[i*n + index] > d_max[index])
				d_max[index] = d_d[i*n + index];
			}
			__syncthreads();
		}
	}
}


int main()
{
	int n,m;
	//cout<<"Enter rows and columns";
	m = 10000, n = 10000;
	int bytes = m*n*sizeof(int);
 
	int *h_a = (int *)malloc(bytes);
	int *h_b = (int *)malloc(bytes);
	int *h_c = (int *)malloc(bytes);
	int *h_d = (int *)malloc(bytes);
	int *h_max = (int *)malloc(n*sizeof(int));
 
	for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
			*(h_a + (i*n + j)) = rand()%2 ;
        }
    }
    
	int *d_a, *d_b, *d_c, *d_d, *d_max;
	cudaMalloc((void**)&d_c, bytes);
	cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_d, bytes);
    cudaMalloc((void**)&d_max, n*sizeof(int));
 
	
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_a, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
	cudaEventCreate( &start);
	cudaEventCreate( &stop);

	cudaEventRecord(start);
    
    downsweep<<<CEIL(n, THREAD), THREAD>>>(d_b, m, n);
	
	sidesweep<<<CEIL(m, THREAD), THREAD>>>(d_c, m, n);
	
	int matrix = m*n;
	maxsquare<<<CEIL(n, THREAD), THREAD>>>(d_d, d_c, d_b, d_max, m, n);
	
	cudaMemcpy(h_max, d_max, n*sizeof(int), cudaMemcpyDeviceToHost);
 
	int gpu_ans = 0;
	for(int i=0; i<n; i++)
	{
		if(*(h_max + i) > gpu_ans)
			gpu_ans = *(h_max + i);
	}
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	
	float milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop);
	
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime = 0;
	cpu_startTime = clock();
	int ans = 1;
	
	//CPU N^2 Method
	for(int i = 1; i < m; i++)
	{
		for(int j = 1; j < n; j++)
		{
			if(*(h_a + (i*n+ j)) == 1)
			{
				int temp = *(h_a + (i*n + j - 1));
				if(*(h_a + ((i-1)*n + j - 1)) < temp)
				temp = *(h_a + ((i-1)*n + j - 1));
				if(*(h_a + ((i-1)*n + j))  < temp)
				temp = *(h_a + ((i-1)*n + j)) ;
				*(h_a + (i*n + j))  = temp + 1;
				if(*(h_a + (i*n + j)) > ans)
				ans = *(h_a + (i*n + j));
			}
		}
	}

	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime) / (1.0 * CLOCKS_PER_SEC)) * 1000;
	
	cout<<"CPU Answer : "<<ans*ans<<"\n";
	cout<<"GPU answer : "<<gpu_ans*gpu_ans<<"\n";
	cout<<"Time taken by CPU : "<<cpu_ElapseTime<<"\n";
	cout<<"Time taken by GPU : "<<milliseconds<<"\n";
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	cudaFree(d_max);
}


