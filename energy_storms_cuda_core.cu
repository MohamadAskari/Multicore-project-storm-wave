#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "energy_storms.h"

__global__ void bombardment_kernel(float *layer, int layer_size, int storm_size, int *storm_posval) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < layer_size) {
        float cell_value = layer[k];

        /* For each particle */
        for (int j = 0; j < storm_size; j++) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storm_posval[j * 2 + 1] * 1000.0f;
            /* Get impact position */
            int position = storm_posval[j * 2];

            float base = energy / (float)layer_size;
            float thresh = THRESHOLD / (float)layer_size;

            /* 1. Compute the absolute value of the distance between the
            impact position and the k-th position of the layer */            int distance = position - k;
            if (distance < 0) distance = -distance;

            /* 2. Impact cell has a distance value of 1 */
            distance = distance + 1;

            /* 3. Square root of the distance */
            /* NOTE: Real world atenuation typically depends on the square of the distance.
            We use here a tailored equation that affects a much wider range of cells */
            float atenuacion = sqrtf((float)distance);

            /* 4. Compute attenuated energy */
            float energy_k = base / atenuacion;

            /* 5. Do not add if its absolute value is lower than the threshold */
            if (energy_k >= thresh || energy_k <= -thresh)
                cell_value += energy_k;
        }

        layer[k] = cell_value;
    }
}

__global__ void relaxation_kernel(float *layer, float *layer_copy, int layer_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k > 0 && k < layer_size - 1) {
        layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
    }
}

__global__ void find_local_maxima_kernel(float *layer, int layer_size, float *max_values, int *max_positions, int num_candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_candidates) {
        int k = idx + 1;
        if (k < layer_size - 1) {
            if (layer[k] > layer[k-1] && layer[k] > layer[k+1]) {
                max_values[idx] = layer[k];
                max_positions[idx] = k;
            } else {
                max_values[idx] = -1.0f;
                max_positions[idx] = -1;
            }
        }
    }
}

__global__ void reduce_max_kernel(float *values, int *positions, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = (n + 1) / 2;
    
    if (idx < stride && (idx + stride) < n) {
        if (values[idx + stride] > values[idx]) {
            values[idx] = values[idx + stride];
            positions[idx] = positions[idx + stride];
        }
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions, int block_size) {
    int i;
   
    float *d_layer, *d_layer_copy;
    cudaMalloc((void**)&d_layer, sizeof(float) * layer_size);
    cudaMalloc((void**)&d_layer_copy, sizeof(float) * layer_size);
    
    cudaMemset(d_layer, 0, sizeof(float) * layer_size);
    cudaMemset(d_layer_copy, 0, sizeof(float) * layer_size);
    
    int num_candidates = layer_size - 2;  
    float *d_max_values;
    int *d_max_positions;

    cudaMalloc((void**)&d_max_values, sizeof(float) * num_candidates);
    cudaMalloc((void**)&d_max_positions, sizeof(int) * num_candidates);

    int **d_storm_posval = (int**)malloc(sizeof(int*) * num_storms);
    int *storm_sizes = (int*)malloc(sizeof(int) * num_storms);

    for (i = 0; i < num_storms; i++) {
        storm_sizes[i] = storms[i].size;
        cudaMalloc((void**)&d_storm_posval[i], sizeof(int) * storms[i].size * 2);
        cudaMemcpy(d_storm_posval[i], storms[i].posval, sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);
    }
    
    int numBlocks = (layer_size + block_size - 1) / block_size;
    int numBlocksReduction = (num_candidates + block_size - 1) / block_size;
    
    /* 4. Storms simulation */
    for (i = 0; i < num_storms; i++) {
        
        /* 4.1. Add impacts energies to layer cells */
        bombardment_kernel<<<numBlocks, block_size>>>(d_layer, layer_size, storm_sizes[i], d_storm_posval[i]);
        
        cudaDeviceSynchronize();
        
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy layer to layer_copy */
        cudaMemcpy(d_layer_copy, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice);
        
        /* 4.2.2. Apply relaxation stencil */
        relaxation_kernel<<<numBlocks, block_size>>>(d_layer, d_layer_copy, layer_size);
        cudaDeviceSynchronize();
        
        /* 4.3. Locate the maximum value in the layer */
        find_local_maxima_kernel<<<numBlocksReduction, block_size>>>(d_layer, layer_size, d_max_values, d_max_positions, num_candidates);
        
        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);

        int remaining = num_candidates;
        while (remaining > 1) {
            int half = (remaining + 1) / 2; 
            int reductionBlocks = (half + block_size - 1) / block_size;
            reduce_max_kernel<<<reductionBlocks, block_size>>>(d_max_values, d_max_positions, remaining);
            cudaDeviceSynchronize();
            remaining = half;
        }
        
        cudaMemcpy(&maximum[i], d_max_values, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&positions[i], d_max_positions, sizeof(int), cudaMemcpyDeviceToHost);
        gettimeofday(&t_end, NULL);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        printf("[Timing] memcpy: %.3f ms\n", elapsed);
    }
    
    cudaFree(d_layer);
    cudaFree(d_layer_copy);
    cudaFree(d_max_values);
    cudaFree(d_max_positions);

    for (i = 0; i < num_storms; i++) {
        cudaFree(d_storm_posval[i]);
    }
    free(d_storm_posval);
    free(storm_sizes);

}