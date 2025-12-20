#include <helper_cuda.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <texture.cuh>

#include <omp.h>

#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM *THREADS_PER_DIM

#include "kmeans_cuda_kernel_texture.cu"


#define BLOCK_DELTA_REDUCE
#define BLOCK_CENTER_REDUCE

// disable CPU-side reductions when using block/GPU reductions
// #define CPU_DELTA_REDUCE
// #define CPU_CENTER_REDUCE

extern "C" int setup(int argc, char **argv); /* function prototype */

// GLOBAL!!!!!
unsigned int num_threads_perdim =
    THREADS_PER_DIM; /* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM; /* temporary */
unsigned int num_threads =
    num_threads_perdim * num_threads_perdim; /* number of threads */
unsigned int num_blocks =
    num_blocks_perdim * num_blocks_perdim; /* number of blocks */

/* _d denotes it resides on the device */
int *membership_new;      /* newly assignment membership */
float *feature_d;         /* inverted data array */
/* original (not inverted) data array */
wrap::cuda::TextureObject<float> features_flipped_d;
int *membership_d;        /* membership on the device */
float *block_new_centers; /* sum of points in a cluster (per block) */
float *clusters_d;        /* cluster centers on the device */
float *block_clusters_d;  /* per block calculation of cluster centers */
int *block_deltas_d;      /* per block calculation of deltas */


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert
 * the data array */
extern "C" void allocateMemory(int npoints, int nfeatures, int nclusters,
                               float **features) {
    num_blocks = npoints / num_threads;
    if (npoints % num_threads > 0) /* defeat truncation */
        num_blocks++;

    num_blocks_perdim = sqrt((double)num_blocks);
    while (num_blocks_perdim * num_blocks_perdim <
           num_blocks) // defeat truncation (should run once)
        num_blocks_perdim++;

    num_blocks = num_blocks_perdim * num_blocks_perdim;

    /* allocate memory for memory_new[] and initialize to -1 (host) */
    membership_new = (int *)malloc(npoints * sizeof(int));
    for (int i = 0; i < npoints; i++) {
        membership_new[i] = -1;
    }

    /* allocate memory for block_new_centers[] (host) */
    block_new_centers = (float *)malloc(nclusters * nfeatures * sizeof(float));

    /* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
    checkCudaErrors(wrap::cuda::malloc2DTextureObject(&features_flipped_d,
                                                      nfeatures, npoints));
    checkCudaErrors(wrap::cuda::memcpy2DTextureObject(&features_flipped_d, features[0],
                                                      nfeatures, npoints));
    checkCudaErrors(
        cudaMalloc((void **)&feature_d, npoints * nfeatures * sizeof(float)));

    /* invert the data array (kernel execution) */
    invert_mapping<<<num_blocks, num_threads>>>(features_flipped_d.tex,
                                                feature_d,
                                                npoints, nfeatures);

    /* allocate memory for membership_d[] and clusters_d[][] (device) */
    checkCudaErrors(cudaMalloc((void **)&membership_d, npoints * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&clusters_d,
                               nclusters * nfeatures * sizeof(float)));


#ifdef BLOCK_DELTA_REDUCE
    // allocate array to hold the per block deltas on the gpu side

    checkCudaErrors(
        cudaMalloc((void **)&block_deltas_d,
                   num_blocks_perdim * num_blocks_perdim * sizeof(int)));
// cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
    // allocate memory and copy to card cluster  array in which to accumulate
    // center points for the next iteration
    cudaMalloc((void **)&block_clusters_d, num_blocks_perdim *
                                               num_blocks_perdim * nclusters *
                                               nfeatures * sizeof(float));
// cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float),
// cudaMemcpyHostToDevice);
#endif
}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C" void deallocateMemory() {
    free(membership_new);
    free(block_new_centers);
    checkCudaErrors(cudaFree(feature_d));
    checkCudaErrors(wrap::cuda::freeTextureObject(&features_flipped_d));
    checkCudaErrors(cudaFree(membership_d));

    checkCudaErrors(cudaFree(clusters_d));
#ifdef BLOCK_CENTER_REDUCE
    checkCudaErrors(cudaFree(block_clusters_d));
#endif
#ifdef BLOCK_DELTA_REDUCE
    checkCudaErrors(cudaFree(block_deltas_d));
#endif
}
/* -------------- deallocateMemory() end ------------------- */


////////////////////////////////////////////////////////////////////////////////
// Program main
// //

int main(int argc, char **argv) {
    // as done in the CUDA start/help document provided
    setup(argc, argv);
}

//																			  //
////////////////////////////////////////////////////////////////////////////////


/* ------------------- kmeansCuda() ------------------------ */
extern "C" int // delta -- had problems when return value was of float type
    kmeansCuda(float **feature,      /* in: [npoints][nfeatures] */
               int nfeatures,        /* number of attributes for each point */
               int npoints,          /* number of data points */
               int nclusters,        /* number of clusters */
               int *membership,      /* which cluster the point belongs to */
               float **clusters,     /* coordinates of cluster centers */
               int *new_centers_len, /* number of elements in each cluster */
               float **new_centers   /* sum of elements in each cluster */
               ) {
    int delta = 0; /* if point has moved */
    int i, j;      /* counters */

    /* copy membership (host to device) */
    checkCudaErrors(cudaMemcpy(membership_d, membership_new,
                               npoints * sizeof(int), cudaMemcpyHostToDevice));

    /* copy clusters (host to device) */
    checkCudaErrors(cudaMemcpy(clusters_d, clusters[0],
                               nclusters * nfeatures * sizeof(float),
                               cudaMemcpyHostToDevice));

    /* set up texture objects */
    cudaTextureObject_t t_features_obj = 0;
    cudaTextureObject_t t_clusters_obj = 0;

    cudaResourceDesc resDesc0;
    memset(&resDesc0, 0, sizeof(resDesc0));
    resDesc0.resType = cudaResourceTypeLinear;
    resDesc0.res.linear.devPtr = feature_d;
    resDesc0.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc0.res.linear.desc.x = 32; // float is 32 bits
    resDesc0.res.linear.sizeInBytes = npoints * nfeatures * sizeof(float);

    cudaTextureDesc texDesc0;
    memset(&texDesc0, 0, sizeof(texDesc0));
    texDesc0.readMode = cudaReadModeElementType;
    texDesc0.filterMode = cudaFilterModePoint;
    texDesc0.normalizedCoords = 0;

    checkCudaErrors(cudaCreateTextureObject(&t_features_obj, &resDesc0, &texDesc0, NULL));

    cudaResourceDesc resDesc2;
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = cudaResourceTypeLinear;
    resDesc2.res.linear.devPtr = clusters_d;
    resDesc2.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc2.res.linear.desc.x = 32;
    resDesc2.res.linear.sizeInBytes = nclusters * nfeatures * sizeof(float);

    cudaTextureDesc texDesc2;
    memset(&texDesc2, 0, sizeof(texDesc2));
    texDesc2.readMode = cudaReadModeElementType;
    texDesc2.filterMode = cudaFilterModePoint;
    texDesc2.normalizedCoords = 0;

    checkCudaErrors(cudaCreateTextureObject(&t_clusters_obj, &resDesc2, &texDesc2, NULL));

    /* copy clusters to constant memory */
    checkCudaErrors(cudaMemcpyToSymbol(c_clusters, clusters[0],
                                       nclusters * nfeatures * sizeof(float), 0,
                                       cudaMemcpyHostToDevice));


    /* setup execution parameters.
           changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3 grid(num_blocks_perdim, num_blocks_perdim);
    dim3 threads(num_threads_perdim * num_threads_perdim);

    /* execute the kernel */
    kmeansPoint<<<grid, threads>>>(feature_d, nfeatures, npoints, nclusters,
                                   membership_d, clusters_d, block_clusters_d,
                                   block_deltas_d,
                                   t_features_obj, features_flipped_d.tex, t_clusters_obj);

    checkCudaErrors(cudaThreadSynchronize());

    checkCudaErrors(cudaDestroyTextureObject(t_features_obj));
    checkCudaErrors(cudaDestroyTextureObject(t_clusters_obj));

    /* copy back membership (device to host) */
    cudaMemcpy(membership_new, membership_d, npoints * sizeof(int),
               cudaMemcpyDeviceToHost);

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float *block_clusters_h =
        (float *)malloc(num_blocks_perdim * num_blocks_perdim * nclusters *
                        nfeatures * sizeof(float));

    checkCudaErrors(cudaMemcpy(block_clusters_h, block_clusters_d,
               num_blocks_perdim * num_blocks_perdim * nclusters * nfeatures *
                   sizeof(float), cudaMemcpyDeviceToHost));
#endif
#ifdef BLOCK_DELTA_REDUCE
    int *block_deltas_h =
        (int *)malloc(num_blocks_perdim * num_blocks_perdim * sizeof(int));

    checkCudaErrors(cudaMemcpy(block_deltas_h, block_deltas_d,
               num_blocks_perdim * num_blocks_perdim * sizeof(int), cudaMemcpyDeviceToHost));
#endif

    /* for each point, sum data points in each cluster
       and see if membership has changed:
         if so, increase delta and change old membership, and update
       new_centers;
         otherwise, update new_centers */
    delta = 0;
    for (i = 0; i < npoints; i++) {
        int cluster_id = membership_new[i];
        new_centers_len[cluster_id]++;
        if (membership_new[i] != membership[i]) {
#ifdef CPU_DELTA_REDUCE
            delta++;
#endif
            membership[i] = membership_new[i];
        }
#ifdef CPU_CENTER_REDUCE
        for (j = 0; j < nfeatures; j++) {
            new_centers[cluster_id][j] += feature[i][j];
        }
#endif
    }


#ifdef BLOCK_DELTA_REDUCE
    /*** calculate global sums from per block sums for delta and the new centers
     * ***/

    // debug
    // printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim
    // * num_blocks_perdim);
    for (i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
        // printf("block %d delta is %d \n",i,block_deltas_h[i]);
        delta += block_deltas_h[i];
    }

#endif
#ifdef BLOCK_CENTER_REDUCE

    for (int j = 0; j < nclusters; j++) {
        for (int k = 0; k < nfeatures; k++) {
            block_new_centers[j * nfeatures + k] = 0.f;
        }
    }

    for (i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
        for (int j = 0; j < nclusters; j++) {
            for (int k = 0; k < nfeatures; k++) {
                block_new_centers[j * nfeatures + k] +=
                    block_clusters_h[i * nclusters * nfeatures + j * nfeatures +
                                     k];
            }
        }
    }


#ifdef CPU_CENTER_REDUCE
// debug
/*for(int j = 0; j < nclusters;j++) {
        for(int k = 0; k < nfeatures;k++) {
                if(new_centers[j][k] >	1.001 *
block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 *
block_new_centers[j*nfeatures + k]) {
                        printf("\t \t for %d:%d, normal value is %e and gpu
reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures +
k]);
                }
        }
}*/
#endif

#ifdef BLOCK_CENTER_REDUCE
    for (int j = 0; j < nclusters; j++) {
        for (int k = 0; k < nfeatures; k++)
            new_centers[j][k] = block_new_centers[j * nfeatures + k];
    }
#endif

#endif

    return delta;
}
/* ------------------- kmeansCuda() end ------------------------ */

/* vim: set ts=4 sw=4 sts=4 et ai: */
