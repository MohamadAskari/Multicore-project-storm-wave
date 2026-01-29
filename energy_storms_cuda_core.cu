#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "energy_storms.h"

#define BLOCK_SIZE 256

/* CUDA kernel to update energy for all cells based on one particle impact with sqrt table */
__global__ void update_energy_kernel(float *layer, int layer_size, int position, float base, float thresh, float *sqrt_table) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < layer_size) {
        /* 1. Compute the absolute value of the distance */
        int distance = position - k;
        if (distance < 0) distance = -distance;
        
        /* 2. Impact cell has a distance value of 1 */
        distance = distance + 1;
        
        /* 3. Square root of the distance - use precomputed table */
        float atenuacion = sqrt_table[distance];
        
        /* 4. Compute attenuated energy */
        float energy_k = base / atenuacion;
        
        /* 5. Do not add if its absolute value is lower than the threshold */
        if (energy_k >= thresh || energy_k <= -thresh)
            atomicAdd(&layer[k], energy_k);
    }
}

/* CUDA kernel for relaxation stencil */
__global__ void relaxation_kernel(float *layer, float *layer_copy, int layer_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k > 0 && k < layer_size - 1) {
        layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
    }
}

/* CUDA kernel to find local maxima */
__global__ void find_local_maxima_kernel(float *layer, int layer_size, float *max_values, int *max_positions, int num_candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_candidates) {
        int k = idx + 1;  // Start from position 1
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

/* Optimized reduction to find maximum on CPU from candidates */
void find_global_maximum(float *max_values, int *max_positions, int num_candidates, float *maximum, int *position) {
    float max_val = -1.0f;
    int max_pos = -1;
    
    for (int i = 0; i < num_candidates; i++) {
        if (max_values[i] > max_val) {
            max_val = max_values[i];
            max_pos = max_positions[i];
        }
    }
    
    *maximum = max_val;
    *position = max_pos;
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j;
    
    /* Find maximum distance for sqrt table */
    int max_distance = layer_size;
    for (i = 0; i < num_storms; i++) {
        for (j = 0; j < storms[i].size; j++) {
            int pos = storms[i].posval[j * 2];
            if (pos + 1 > max_distance) max_distance = pos + 1;
        }
    }
    
    /* Precompute sqrt table on host */
    float *h_sqrt_table = (float*)malloc(sizeof(float) * (max_distance + 1));
    h_sqrt_table[0] = 1.0f;  // Avoid division by zero
    for (int d = 1; d <= max_distance; d++) {
        h_sqrt_table[d] = sqrtf((float)d);
    }
    
    /* Allocate device memory for sqrt table and copy */
    float *d_sqrt_table;
    cudaMalloc((void**)&d_sqrt_table, sizeof(float) * (max_distance + 1));
    cudaMemcpy(d_sqrt_table, h_sqrt_table, sizeof(float) * (max_distance + 1), cudaMemcpyHostToDevice);
    
    /* Allocate device memory for layers */
    float *d_layer, *d_layer_copy;
    cudaMalloc((void**)&d_layer, sizeof(float) * layer_size);
    cudaMalloc((void**)&d_layer_copy, sizeof(float) * layer_size);
    
    /* Initialize device layer to zero */
    cudaMemset(d_layer, 0, sizeof(float) * layer_size);
    cudaMemset(d_layer_copy, 0, sizeof(float) * layer_size);
    
    /* Allocate memory for reduction results */
    int num_candidates = layer_size - 2;  // Exclude first and last positions
    float *d_max_values, *h_max_values;
    int *d_max_positions, *h_max_positions;
    
    cudaMalloc((void**)&d_max_values, sizeof(float) * num_candidates);
    cudaMalloc((void**)&d_max_positions, sizeof(int) * num_candidates);
    h_max_values = (float*)malloc(sizeof(float) * num_candidates);
    h_max_positions = (int*)malloc(sizeof(int) * num_candidates);
    
    /* Calculate grid dimensions */
    int numBlocks = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksReduction = (num_candidates + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    /* 4. Storms simulation */
    for (i = 0; i < num_storms; i++) {
        
        /* 4.1. Add impacts energies to layer cells */
        for (j = 0; j < storms[i].size; j++) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000.0f;
            /* Get impact position */
            int position = storms[i].posval[j*2];
            
            /* Precompute constants */
            float base = energy / (float)layer_size;
            float thresh = THRESHOLD / (float)layer_size;
            
            /* Launch kernel to update all cells for this particle */
            update_energy_kernel<<<numBlocks, BLOCK_SIZE>>>(d_layer, layer_size, position, base, thresh, d_sqrt_table);
        }
        
        /* Synchronize after all particles in the storm */
        cudaDeviceSynchronize();
        
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy layer to layer_copy */
        cudaMemcpy(d_layer_copy, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice);
        
        /* 4.2.2. Apply relaxation stencil */
        relaxation_kernel<<<numBlocks, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
        cudaDeviceSynchronize();
        
        /* 4.3. Locate the maximum value in the layer */
        /* Find all local maxima */
        find_local_maxima_kernel<<<numBlocksReduction, BLOCK_SIZE>>>(d_layer, layer_size, d_max_values, d_max_positions, num_candidates);
        
        /* Copy candidates to host */
        cudaMemcpy(h_max_values, d_max_values, sizeof(float) * num_candidates, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max_positions, d_max_positions, sizeof(int) * num_candidates, cudaMemcpyDeviceToHost);
        
        /* Find global maximum on CPU */
        find_global_maximum(h_max_values, h_max_positions, num_candidates, &maximum[i], &positions[i]);
    }
    
    /* Free device memory */
    cudaFree(d_layer);
    cudaFree(d_layer_copy);
    cudaFree(d_max_values);
    cudaFree(d_max_positions);
    cudaFree(d_sqrt_table);
    
    /* Free host memory */
    free(h_max_values);
    free(h_max_positions);
    free(h_sqrt_table);
}