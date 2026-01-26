#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k;
    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );
    float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
    float *sqrt_table = (float *)malloc(sizeof(float) * (layer_size + 1));
    if ( !layer || !layer_copy || !sqrt_table ) {
        fprintf(stderr,"Error: Allocating memory\n");
        exit( EXIT_FAILURE );
    }

    // Precompute sqrt(d) once
    sqrt_table[0] = 0.0f;
    #pragma omp parallel for schedule(static)
    for (int d = 1; d <= layer_size; d++) {
        sqrt_table[d] = sqrtf((float)d);
    }

    for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;
    for( k=0; k<layer_size; k++ ) layer_copy[k] = 0.0f;
    
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

                /* For each cell in the layer */
                // use nowait to avoid waiting for all threads to finish
                #pragma omp for schedule(static) nowait
                for( k=0; k<layer_size; k++ ) {
                    /* 1. Compute the absolute value of the distance between the
                    impact position and the k-th position of the layer */
                    int distance = position - k;
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
                        layer[k] = layer[k] + energy_k;
                }
            }

            // wait for all impacts to finish before relaxation
            #pragma omp barrier

            /* 4.2. Energy relaxation between storms */
            /* 4.2.1. Copy values to the ancillary array */
            for( k=0; k<layer_size; k++ ) 
                layer_copy[k] = layer[k];
            
            #pragma omp barrier
            /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
            for( k=1; k<layer_size-1; k++ )
                layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;


            float local_max = -1.0f; 
            int local_pos = -1;

            /* 4.3. Locate the maximum value in the layer, and its position */
            #pragma omp for schedule(static) nowait
            for( k=1; k<layer_size-1; k++ ) {
                /* Check it only if it is a local maximum */
                if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                    if ( layer[k] > local_max ) {
                        local_max = layer[k];
                        local_pos = k;
                    }
                }
            }

            // safely update global maximum
            #pragma omp critical
            {
                if ( local_max > maximum[i] ) {
                    maximum[i] = local_max;
                    positions[i] = local_pos;
                }
            }
        }
    }

    free(layer);
    free(layer_copy);
    free(sqrt_table);
}