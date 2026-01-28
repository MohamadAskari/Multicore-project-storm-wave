#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "energy_storms.h"

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k;
    
    /* Get MPI rank and size */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Find maximum particle position across all storms to determine sqrt_table size */
    int max_position = layer_size - 1;
    for (i = 0; i < num_storms; i++) {
        for (j = 0; j < storms[i].size; j++) {
            int pos = storms[i].posval[j * 2];
            if (pos > max_position) {
                max_position = pos;
            }
        }
    }
    
    /* Maximum distance can be from position 0 to max_position (or vice versa), plus 1 */
    int max_distance = max_position + 1;
    
    /* Calculate local portion of the layer for this process */
    int chunk_size = layer_size / size;
    int remainder = layer_size % size;
    
    /* Each process gets chunk_size elements, with the first 'remainder' processes getting one extra */
    int local_start = rank * chunk_size + (rank < remainder ? rank : remainder);
    int local_size = chunk_size + (rank < remainder ? 1 : 0);
    int local_end = local_start + local_size;
    
    /* Allocate local layer with halos (one cell on each side for stencil) */
    float *local_layer = (float *)malloc( sizeof(float) * (local_size + 2) );
    float *local_layer_copy = (float *)malloc( sizeof(float) * (local_size + 2) );
    float *sqrt_table = (float *)malloc(sizeof(float) * (max_distance + 1));
    if ( !local_layer || !local_layer_copy || !sqrt_table ) {
        fprintf(stderr,"Error: Allocating memory\n");
        exit( EXIT_FAILURE );
    }

    /* Precompute sqrt(d) once - all processes compute the same table */
    sqrt_table[0] = 0.0f;
    #pragma omp parallel for schedule(static)
    for (int d = 1; d <= max_distance; d++) {
        sqrt_table[d] = sqrtf((float)d);
    }

    /* Initialize local layer (index 0 is left halo, indices 1 to local_size are actual data, local_size+1 is right halo) */
    for( k=0; k<local_size+2; k++ ) local_layer[k] = 0.0f;
    for( k=0; k<local_size+2; k++ ) local_layer_copy[k] = 0.0f;
    
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {

        // create threads once per storm
        #pragma omp parallel private(j, k)
        {
            /* 4.1. Add impacts energies to layer cells */
            /* For each particle */
            for( j=0; j<storms[i].size; j++ ) {
                /* Get impact energy (expressed in thousandths) */
                float energy = (float)storms[i].posval[j*2+1] * 1000;
                /* Get impact position */
                int position = storms[i].posval[j*2];

                // precompute base and threshold
                float base   = energy / (float)layer_size;
                float thresh = THRESHOLD / (float)layer_size;

                /* For each cell in the LOCAL layer portion */
                /* Note: local_layer indices are offset by 1 (index 0 is left halo) */
                #pragma omp for schedule(static) nowait
                for( k=0; k<local_size; k++ ) {
                    /* Global position corresponding to local index k */
                    int global_k = local_start + k;
                    
                    /* 1. Compute the absolute value of the distance between the
                    impact position and the k-th position of the layer */
                    int distance = position - global_k;
                    if ( distance < 0 ) distance = - distance;

                    /* 2. Impact cell has a distance value of 1 */
                    distance = distance + 1;

                    /* 3. Square root of the distance */
                    /* NOTE: Real world atenuation typically depends on the square of the distance.
                    We use here a tailored equation that affects a much wider range of cells */
                    // float atenuacion = sqrtf( (float)distance );

                    /* 4. Compute attenuated energy */
                    float energy_k = base / sqrt_table[distance];

                    /* 5. Do not add if its absolute value is lower than the threshold */
                    if ( energy_k >= thresh || energy_k <= -thresh )
                        local_layer[k+1] = local_layer[k+1] + energy_k;  // k+1 because index 0 is halo
                }
            }

            // wait for all impacts to finish before relaxation
            #pragma omp barrier
        } // end of parallel region for impacts
        
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Exchange halo cells with neighbors */
        MPI_Request requests[4];
        MPI_Status statuses[4];
        int num_requests = 0;
        
        /* Send right boundary to right neighbor, receive left halo from left neighbor */
        if (rank < size - 1) {
            MPI_Isend(&local_layer[local_size], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        }
        if (rank > 0) {
            MPI_Irecv(&local_layer[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        }
        
        /* Send left boundary to left neighbor, receive right halo from right neighbor */
        if (rank > 0) {
            MPI_Isend(&local_layer[1], 1, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        }
        if (rank < size - 1) {
            MPI_Irecv(&local_layer[local_size + 1], 1, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        }
        
        MPI_Waitall(num_requests, requests, statuses);
        
        /* 4.2.2. Copy values to the ancillary array */
        #pragma omp parallel for schedule(static)
        for( k=0; k<local_size+2; k++ ) 
            local_layer_copy[k] = local_layer[k];
        
        /* 4.2.3. Update layer using the ancillary values with stencil */
        /* For interior cells of the local portion */
        #pragma omp parallel for schedule(static)
        for( k=1; k<local_size+1; k++ ) {
            int global_k = local_start + (k - 1);
            /* Skip the very first and last positions of the GLOBAL layer */
            if (global_k > 0 && global_k < layer_size - 1) {
                local_layer[k] = ( local_layer_copy[k-1] + local_layer_copy[k] + local_layer_copy[k+1] ) / 3;
            }
        }


        /* 4.3. Locate the maximum value in the local layer portion, and its position */
        float thread_max = -1.0f;
        int thread_pos = -1;
        
        #pragma omp parallel private(k)
        {
            float local_max = -1.0f;
            int local_pos = -1;
            
            #pragma omp for schedule(static) nowait
            for( k=1; k<local_size+1; k++ ) {
                int global_k = local_start + (k - 1);
                /* Check it only if it is within bounds and is a local maximum */
                if (global_k > 0 && global_k < layer_size - 1) {
                    if ( local_layer[k] > local_layer[k-1] && local_layer[k] > local_layer[k+1] ) {
                        if ( local_layer[k] > local_max ) {
                            local_max = local_layer[k];
                            local_pos = global_k;
                        }
                    }
                }
            }
            
            // safely update thread maximum
            #pragma omp critical
            {
                if ( local_max > thread_max ) {
                    thread_max = local_max;
                    thread_pos = local_pos;
                }
            }
        }
        
        /* 4.4. Reduce across all MPI processes to find global maximum */
        struct {
            float value;
            int rank;
        } local_data, global_data;
        
        local_data.value = thread_max;
        local_data.rank = rank;
        
        /* Find the global maximum value and which rank has it */
        MPI_Allreduce(&local_data, &global_data, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        maximum[i] = global_data.value;
        
        /* Broadcast the position from the rank that has the maximum */
        int global_pos = thread_pos;
        MPI_Bcast(&global_pos, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);
        positions[i] = global_pos;
    }

    free(local_layer);
    free(local_layer_copy);
    free(sqrt_table);
}