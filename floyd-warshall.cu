%%cu
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

__global__ void calculate(long *d_adj, int k, int n){
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = (x * n) + y;
    int idx1 = (x * n) + k;
    int idx2 = (k * n) + y;
    long s = d_adj[idx1] + d_adj[idx2];

    __syncthreads();

    if(s < d_adj[idx])
        d_adj[idx] = s;

    __syncthreads();
}

int main(int argc, char** argv){
    int i, j, k, n = 10;

    long h_adj[n * n], h_ans[n * n];

    printf("Adjaceny Matrix\n\n");
    for (i = 0;i < n; i++){
        for (j = 0;j < n; j++){
            int pos = (i * n) + j;   
            if (i == j) {
                h_adj[pos] = 0;
                printf("%ld ", h_adj[pos]);
                continue;
            }
   
            h_adj[pos] = rand()%n;
            h_adj[pos] += 1;
            printf("%ld ", h_adj[pos]);
        }
        printf("\n");
    }

    long* d_adj;
    cudaMalloc((void**)&d_adj, n * n * sizeof(long*));
    cudaMemcpy(d_adj, h_adj, n * n * sizeof(long), cudaMemcpyHostToDevice);

    //GPU method
    for(k = 0; k < n; k++)
        calculate<<<1, dim3(n, n, 1)>>>(d_adj, k, n);

    cudaMemcpy(h_ans, d_adj, n * n * sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(d_adj);

    printf("\nShortest Pairwise Distances\n\n");
    for (i = 0;i < n; i++){
        for (j = 0;j < n; j++){
            int pos = (i * n) + j;    
            printf("%ld ", h_ans[pos]);
        }
        printf("\n");
    }

    //CPU method
    for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                int idx1 = (i * n) + k;
                int idx2 = (k * n) + j;
                int idx = (i * n) + j;
                long temp = h_adj[idx1] + h_adj[idx2];
                if (temp < h_adj[idx])
                  h_adj[idx] = temp;
            }
        }
    }

    printf("\nCPU answer\n\n");
    int ans_check = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            int idx = (i * n) + j;
            printf("%ld ", h_adj[idx]);
            if (h_adj[idx] != h_ans[idx]) 
                ans_check = 1;
        }
        printf("\n");
    }

    if (ans_check)
      printf("\nWrong Answer");
    else
      printf("\nCorrect Answer");

    return 0;
}
