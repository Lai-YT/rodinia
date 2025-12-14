// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <cassert>
#include <cstddef>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <texture.cuh>


/*
 * Options
 *
 */
#define GAMMA 1.4f
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3 // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

/*
 * not options
 */

#define BLOCK_SIZE 192
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

#define NEL 97046
#define NELR \
    BLOCK_SIZE * ((NEL / BLOCK_SIZE) + std::min(1, NEL % BLOCK_SIZE))

/*
 * Generic functions
 */
template <typename T> T *alloc(int N) {
    T *t;
    checkCudaErrors(cudaMalloc((void **)&t, sizeof(T) * N));
    return t;
}

template <typename T> void dealloc(T *array) {
    checkCudaErrors(cudaFree((void *)array));
}

template <typename T> void copy(T *dst, T *src, int N) {
    checkCudaErrors(cudaMemcpy((void *)dst, (void *)src, N * sizeof(T),
                               cudaMemcpyDeviceToDevice));
}

template <typename T> void upload(T *dst, T *src, int N) {
    checkCudaErrors(cudaMemcpy((void *)dst, (void *)src, N * sizeof(T),
                               cudaMemcpyHostToDevice));
}

template <typename T> void download(T *dst, T *src, int N) {
    checkCudaErrors(cudaMemcpy((void *)dst, (void *)src, N * sizeof(T),
                               cudaMemcpyDeviceToHost));
}

template <typename T>
__host__ __device__
size_t nbytes(int N) {
    return sizeof(T) * N;
}

void dump(wrap::cuda::SurfaceObject<float> variables, int nel, int nelr) {
    float *h_variables = new float[nelr * NVAR];
    checkCudaErrors(cudaMemcpy2DFromArray(
        h_variables, sizeof(float) * nelr,
        variables.devPtr, 0, 0,
        sizeof(float) * nelr, NVAR,
        cudaMemcpyDeviceToHost));

    {
        std::ofstream file("density");
        file << nel << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
    }


    {
        std::ofstream file("momentum");
        file << nel << std::endl;
        for (int i = 0; i < nel; i++) {
            for (int j = 0; j != NDIM; j++)
                file << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
            file << std::endl;
        }
    }

    {
        std::ofstream file("density_energy");
        file << nel << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
    }
    delete[] h_variables;
}

/*
 * Element-based Cell-centered FVM solver functions
 */
__constant__ float ff_variable[NVAR];
__constant__ float3 ff_flux_contribution_momentum_x[1];
__constant__ float3 ff_flux_contribution_momentum_y[1];
__constant__ float3 ff_flux_contribution_momentum_z[1];
__constant__ float3 ff_flux_contribution_density_energy[1];

__global__ void cuda_initialize_variables(int nelr, float *variables) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int j = 0; j < NVAR; j++)
        variables[i + j * nelr] = ff_variable[j];
}
__global__ void cuda_initialize_variables(int nelr, float variables[NVAR][NELR]) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int j = 0; j < NVAR; j++)
        variables[j][i] = ff_variable[j];
}
__global__ void cuda_initialize_variables(int nelr, cudaSurfaceObject_t variables) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int j = 0; j < NVAR; j++)
        surf2Dwrite(ff_variable[j], variables, i * sizeof(float), j);
}
void initialize_variables(int nelr, float *variables) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
    getLastCudaError("initialize_variables failed");
}
void initialize_variables(int nelr, wrap::cuda::SurfaceObject<float> &variables) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_initialize_variables<<<Dg, Db>>>(nelr, variables.surf);
    getLastCudaError("initialize_variables failed");
}

__device__ __host__ inline void compute_flux_contribution(
    float &density, float3 &momentum, float &density_energy, float &pressure,
    float3 &velocity, float3 &fc_momentum_x, float3 &fc_momentum_y,
    float3 &fc_momentum_z, float3 &fc_density_energy) {
    fc_momentum_x.x = velocity.x * momentum.x + pressure;
    fc_momentum_x.y = velocity.x * momentum.y;
    fc_momentum_x.z = velocity.x * momentum.z;


    fc_momentum_y.x = fc_momentum_x.y;
    fc_momentum_y.y = velocity.y * momentum.y + pressure;
    fc_momentum_y.z = velocity.y * momentum.z;

    fc_momentum_z.x = fc_momentum_x.z;
    fc_momentum_z.y = fc_momentum_y.z;
    fc_momentum_z.z = velocity.z * momentum.z + pressure;

    float de_p = density_energy + pressure;
    fc_density_energy.x = velocity.x * de_p;
    fc_density_energy.y = velocity.y * de_p;
    fc_density_energy.z = velocity.z * de_p;
}

__device__ inline void compute_velocity(float &density, float3 &momentum,
                                        float3 &velocity) {
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

__device__ inline float compute_speed_sqd(float3 &velocity) {
    return velocity.x * velocity.x + velocity.y * velocity.y +
           velocity.z * velocity.z;
}

__device__ inline float compute_pressure(float &density, float &density_energy,
                                         float &speed_sqd) {
    return (float(GAMMA) - float(1.0f)) *
           (density_energy - float(0.5f) * density * speed_sqd);
}

__device__ inline float compute_speed_of_sound(float &density,
                                               float &pressure) {
    return sqrtf(float(GAMMA) * pressure / density);
}

__global__ void cuda_compute_step_factor(int nelr, cudaSurfaceObject_t variables,
                                         float *areas, float *step_factors) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);

    float density = surf2Dread<float>(variables, i * sizeof(float), VAR_DENSITY);
    float3 momentum;
    momentum.x = surf2Dread<float>(variables, nbytes<float>(i), VAR_MOMENTUM + 0);
    momentum.y = surf2Dread<float>(variables, nbytes<float>(i), VAR_MOMENTUM + 1);
    momentum.z = surf2Dread<float>(variables, nbytes<float>(i), VAR_MOMENTUM + 2);

    float density_energy = surf2Dread<float>(variables, nbytes<float>(i), VAR_DENSITY_ENERGY);

    float3 velocity;
    compute_velocity(density, momentum, velocity);
    float speed_sqd = compute_speed_sqd(velocity);
    float pressure = compute_pressure(density, density_energy, speed_sqd);
    float speed_of_sound = compute_speed_of_sound(density, pressure);

    // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time
    // stepping, this later would need to be divided by the area, so we just do
    // it all at once
    step_factors[i] =
        float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}
void compute_step_factor(int nelr, wrap::cuda::SurfaceObject<float> variables, float *areas,
                         float *step_factors) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables.surf, areas, step_factors);
    getLastCudaError("compute_step_factor failed");
}

/*
 *
 *
*/
__global__ void cuda_compute_flux(int nelr, int *elements_surrounding_elements,
                                  float *normals, cudaSurfaceObject_t variables,
                                  float *fluxes) {
    const float smoothing_coefficient = float(0.2f);
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);

    int j, nb;
    float3 normal;
    float normal_len;
    float factor;

    float density_i = surf2Dread<float>(variables, nbytes<float>(i), VAR_DENSITY);
    float3 momentum_i;
    momentum_i.x = surf2Dread<float>(variables, nbytes<float>(i), VAR_MOMENTUM + 0);
    momentum_i.y = surf2Dread<float>(variables, nbytes<float>(i), VAR_MOMENTUM + 1);
    momentum_i.z = surf2Dread<float>(variables, nbytes<float>(i), VAR_MOMENTUM + 2);

    float density_energy_i = surf2Dread<float>(variables, nbytes<float>(i), VAR_DENSITY_ENERGY);

    float3 velocity_i;
    compute_velocity(density_i, momentum_i, velocity_i);
    float speed_sqd_i = compute_speed_sqd(velocity_i);
    float speed_i = sqrtf(speed_sqd_i);
    float pressure_i =
        compute_pressure(density_i, density_energy_i, speed_sqd_i);
    float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
    float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
        flux_contribution_i_momentum_z;
    float3 flux_contribution_i_density_energy;
    compute_flux_contribution(
        density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
        flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
        flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

    float flux_i_density = float(0.0f);
    float3 flux_i_momentum;
    flux_i_momentum.x = float(0.0f);
    flux_i_momentum.y = float(0.0f);
    flux_i_momentum.z = float(0.0f);
    float flux_i_density_energy = float(0.0f);

    float3 velocity_nb;
    float density_nb, density_energy_nb;
    float3 momentum_nb;
    float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y,
        flux_contribution_nb_momentum_z;
    float3 flux_contribution_nb_density_energy;
    float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

#pragma unroll
    for (j = 0; j < NNB; j++) {
        nb = elements_surrounding_elements[i + j * nelr];
        normal.x = normals[i + (j + 0 * NNB) * nelr];
        normal.y = normals[i + (j + 1 * NNB) * nelr];
        normal.z = normals[i + (j + 2 * NNB) * nelr];
        normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y +
                           normal.z * normal.z);

        if (nb >= 0) // a legitimate neighbor
        {
            density_nb = surf2Dread<float>(variables, nbytes<float>(nb), VAR_DENSITY);
            momentum_nb.x = surf2Dread<float>(variables, nbytes<float>(nb), VAR_MOMENTUM + 0);
            momentum_nb.y = surf2Dread<float>(variables, nbytes<float>(nb), VAR_MOMENTUM + 1);
            momentum_nb.z = surf2Dread<float>(variables, nbytes<float>(nb), VAR_MOMENTUM + 2);
            density_energy_nb = surf2Dread<float>(variables, nbytes<float>(nb), VAR_DENSITY_ENERGY);
            compute_velocity(density_nb, momentum_nb, velocity_nb);
            speed_sqd_nb = compute_speed_sqd(velocity_nb);
            pressure_nb =
                compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
            speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
            compute_flux_contribution(
                density_nb, momentum_nb, density_energy_nb, pressure_nb,
                velocity_nb, flux_contribution_nb_momentum_x,
                flux_contribution_nb_momentum_y,
                flux_contribution_nb_momentum_z,
                flux_contribution_nb_density_energy);

            // artificial viscosity
            factor = -normal_len * smoothing_coefficient * float(0.5f) *
                     (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i +
                      speed_of_sound_nb);
            flux_i_density += factor * (density_i - density_nb);
            flux_i_density_energy +=
                factor * (density_energy_i - density_energy_nb);
            flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
            flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
            flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

            // accumulate cell-centered fluxes
            factor = float(0.5f) * normal.x;
            flux_i_density += factor * (momentum_nb.x + momentum_i.x);
            flux_i_density_energy +=
                factor * (flux_contribution_nb_density_energy.x +
                          flux_contribution_i_density_energy.x);
            flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.x +
                                           flux_contribution_i_momentum_x.x);
            flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.x +
                                           flux_contribution_i_momentum_y.x);
            flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.x +
                                           flux_contribution_i_momentum_z.x);

            factor = float(0.5f) * normal.y;
            flux_i_density += factor * (momentum_nb.y + momentum_i.y);
            flux_i_density_energy +=
                factor * (flux_contribution_nb_density_energy.y +
                          flux_contribution_i_density_energy.y);
            flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.y +
                                           flux_contribution_i_momentum_x.y);
            flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.y +
                                           flux_contribution_i_momentum_y.y);
            flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.y +
                                           flux_contribution_i_momentum_z.y);

            factor = float(0.5f) * normal.z;
            flux_i_density += factor * (momentum_nb.z + momentum_i.z);
            flux_i_density_energy +=
                factor * (flux_contribution_nb_density_energy.z +
                          flux_contribution_i_density_energy.z);
            flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.z +
                                           flux_contribution_i_momentum_x.z);
            flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.z +
                                           flux_contribution_i_momentum_y.z);
            flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.z +
                                           flux_contribution_i_momentum_z.z);
        } else if (nb == -1) // a wing boundary
        {
            flux_i_momentum.x += normal.x * pressure_i;
            flux_i_momentum.y += normal.y * pressure_i;
            flux_i_momentum.z += normal.z * pressure_i;
        } else if (nb == -2) // a far field boundary
        {
            factor = float(0.5f) * normal.x;
            flux_i_density +=
                factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
            flux_i_density_energy +=
                factor * (ff_flux_contribution_density_energy[0].x +
                          flux_contribution_i_density_energy.x);
            flux_i_momentum.x +=
                factor * (ff_flux_contribution_momentum_x[0].x +
                          flux_contribution_i_momentum_x.x);
            flux_i_momentum.y +=
                factor * (ff_flux_contribution_momentum_y[0].x +
                          flux_contribution_i_momentum_y.x);
            flux_i_momentum.z +=
                factor * (ff_flux_contribution_momentum_z[0].x +
                          flux_contribution_i_momentum_z.x);

            factor = float(0.5f) * normal.y;
            flux_i_density +=
                factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
            flux_i_density_energy +=
                factor * (ff_flux_contribution_density_energy[0].y +
                          flux_contribution_i_density_energy.y);
            flux_i_momentum.x +=
                factor * (ff_flux_contribution_momentum_x[0].y +
                          flux_contribution_i_momentum_x.y);
            flux_i_momentum.y +=
                factor * (ff_flux_contribution_momentum_y[0].y +
                          flux_contribution_i_momentum_y.y);
            flux_i_momentum.z +=
                factor * (ff_flux_contribution_momentum_z[0].y +
                          flux_contribution_i_momentum_z.y);

            factor = float(0.5f) * normal.z;
            flux_i_density +=
                factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
            flux_i_density_energy +=
                factor * (ff_flux_contribution_density_energy[0].z +
                          flux_contribution_i_density_energy.z);
            flux_i_momentum.x +=
                factor * (ff_flux_contribution_momentum_x[0].z +
                          flux_contribution_i_momentum_x.z);
            flux_i_momentum.y +=
                factor * (ff_flux_contribution_momentum_y[0].z +
                          flux_contribution_i_momentum_y.z);
            flux_i_momentum.z +=
                factor * (ff_flux_contribution_momentum_z[0].z +
                          flux_contribution_i_momentum_z.z);
        }
    }

    fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
    fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
    fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
    fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
    fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
}
void compute_flux(int nelr, int *elements_surrounding_elements, float *normals,
                  wrap::cuda::SurfaceObject<float> variables, float *fluxes) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_compute_flux<<<Dg, Db>>>(nelr, elements_surrounding_elements, normals,
                                  variables.surf, fluxes);
    getLastCudaError("compute_flux failed");
}

__global__ void cuda_time_step(int j, int nelr, cudaSurfaceObject_t old_variables,
                               cudaSurfaceObject_t variables, float *step_factors,
                               float *fluxes) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);

    float factor = step_factors[i] / float(RK + 1 - j);

    float old_density = surf2Dread<float>(old_variables, nbytes<float>(i), VAR_DENSITY);
    surf2Dwrite(old_density + factor * fluxes[i + VAR_DENSITY * nelr],
                variables, nbytes<float>(i), VAR_DENSITY);
    float old_density_energy = surf2Dread<float>(old_variables, nbytes<float>(i), VAR_DENSITY_ENERGY);
    surf2Dwrite(old_density_energy + factor * fluxes[i + VAR_DENSITY_ENERGY * nelr],
                variables, nbytes<float>(i), VAR_DENSITY_ENERGY);
    float old_momentum_x = surf2Dread<float>(old_variables, nbytes<float>(i), VAR_MOMENTUM + 0);
    surf2Dwrite(old_momentum_x + factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr],
                variables, nbytes<float>(i), VAR_MOMENTUM + 0);
    float old_momentum_y = surf2Dread<float>(old_variables, nbytes<float>(i), VAR_MOMENTUM + 1);
    surf2Dwrite(old_momentum_y + factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr],
                variables, nbytes<float>(i), VAR_MOMENTUM + 1);
    float old_momentum_z = surf2Dread<float>(old_variables, nbytes<float>(i), VAR_MOMENTUM + 2);
    surf2Dwrite(old_momentum_z + factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr],
                variables, nbytes<float>(i), VAR_MOMENTUM + 2);
}
void time_step(int j, int nelr, wrap::cuda::SurfaceObject<float> old_variables,
               wrap::cuda::SurfaceObject<float> variables,
               float *step_factors, float *fluxes) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_time_step<<<Dg, Db>>>(j, nelr, old_variables.surf, variables.surf, step_factors,
                               fluxes);
    getLastCudaError("update failed");
}

/*
 * Main function
 */
int main(int argc, char **argv) {
    printf("WG size of kernel:initialize = %d, WG size of "
           "kernel:compute_step_factor = %d, WG size of kernel:compute_flux = "
           "%d, WG size of kernel:time_step = %d\n",
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    if (argc < 2) {
        std::cout << "specify data file name" << std::endl;
        return 0;
    }
    const char *data_file_name = argv[1];

    cudaDeviceProp prop;
    int dev;

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

    printf("Name:                     %s\n", prop.name);

    // set far field conditions and load them into constant memory on the gpu
    {
        float h_ff_variable[NVAR];
        const float angle_of_attack =
            float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

        h_ff_variable[VAR_DENSITY] = float(1.4);

        float ff_pressure = float(1.0f);
        float ff_speed_of_sound =
            sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
        float ff_speed = float(ff_mach) * ff_speed_of_sound;

        float3 ff_velocity;
        ff_velocity.x = ff_speed * float(cos((float)angle_of_attack));
        ff_velocity.y = ff_speed * float(sin((float)angle_of_attack));
        ff_velocity.z = 0.0f;

        h_ff_variable[VAR_MOMENTUM + 0] =
            h_ff_variable[VAR_DENSITY] * ff_velocity.x;
        h_ff_variable[VAR_MOMENTUM + 1] =
            h_ff_variable[VAR_DENSITY] * ff_velocity.y;
        h_ff_variable[VAR_MOMENTUM + 2] =
            h_ff_variable[VAR_DENSITY] * ff_velocity.z;

        h_ff_variable[VAR_DENSITY_ENERGY] =
            h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) +
            (ff_pressure / float(GAMMA - 1.0f));

        float3 h_ff_momentum;
        h_ff_momentum.x = *(h_ff_variable + VAR_MOMENTUM + 0);
        h_ff_momentum.y = *(h_ff_variable + VAR_MOMENTUM + 1);
        h_ff_momentum.z = *(h_ff_variable + VAR_MOMENTUM + 2);
        float3 h_ff_flux_contribution_momentum_x;
        float3 h_ff_flux_contribution_momentum_y;
        float3 h_ff_flux_contribution_momentum_z;
        float3 h_ff_flux_contribution_density_energy;
        compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum,
                                  h_ff_variable[VAR_DENSITY_ENERGY],
                                  ff_pressure, ff_velocity,
                                  h_ff_flux_contribution_momentum_x,
                                  h_ff_flux_contribution_momentum_y,
                                  h_ff_flux_contribution_momentum_z,
                                  h_ff_flux_contribution_density_energy);

        // copy far field conditions to the gpu
        checkCudaErrors(cudaMemcpyToSymbol(ff_variable, h_ff_variable,
                                           NVAR * sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_x,
                                           &h_ff_flux_contribution_momentum_x,
                                           sizeof(float3)));
        checkCudaErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_y,
                                           &h_ff_flux_contribution_momentum_y,
                                           sizeof(float3)));
        checkCudaErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_z,
                                           &h_ff_flux_contribution_momentum_z,
                                           sizeof(float3)));

        checkCudaErrors(cudaMemcpyToSymbol(
            ff_flux_contribution_density_energy,
            &h_ff_flux_contribution_density_energy, sizeof(float3)));
    }

    // read in domain geometry
    float *areas;
    int *elements_surrounding_elements;
    float *normals;
    {
        std::ifstream file(data_file_name);

        int tmp;
        file >> tmp;
        assert(tmp == NEL);

        float *h_areas = new float[NELR];
        int *h_elements_surrounding_elements = new int[NELR * NNB];
        float *h_normals = new float[NELR * NDIM * NNB];


        // read in data
        for (int i = 0; i < NEL; i++) {
            file >> h_areas[i];
            for (int j = 0; j < NNB; j++) {
                file >> h_elements_surrounding_elements[i + j * NELR];
                if (h_elements_surrounding_elements[i + j * NELR] < 0)
                    h_elements_surrounding_elements[i + j * NELR] = -1;
                h_elements_surrounding_elements[i + j * NELR]--; // it's coming
                                                                 // in with
                                                                 // Fortran
                                                                 // numbering

                for (int k = 0; k < NDIM; k++) {
                    file >> h_normals[i + (j + k * NNB) * NELR];
                    h_normals[i + (j + k * NNB) * NELR] =
                        -h_normals[i + (j + k * NNB) * NELR];
                }
            }
        }

        // fill in remaining data
        int last = NEL - 1;
        for (int i = NEL; i < NELR; i++) {
            h_areas[i] = h_areas[last];
            for (int j = 0; j < NNB; j++) {
                // duplicate the last element
                h_elements_surrounding_elements[i + j * NELR] =
                    h_elements_surrounding_elements[last + j * NELR];
                for (int k = 0; k < NDIM; k++)
                    h_normals[last + (j + k * NNB) * NELR] =
                        h_normals[last + (j + k * NNB) * NELR];
            }
        }

        areas = alloc<float>(NELR);
        upload<float>(areas, h_areas, NELR);

        elements_surrounding_elements = alloc<int>(NELR * NNB);
        upload<int>(elements_surrounding_elements,
                    h_elements_surrounding_elements, NELR * NNB);

        normals = alloc<float>(NELR * NDIM * NNB);
        upload<float>(normals, h_normals, NELR * NDIM * NNB);

        delete[] h_areas;
        delete[] h_elements_surrounding_elements;
        delete[] h_normals;
    }

    // Create arrays and set initial conditions
    wrap::cuda::SurfaceObject<float> variables;
    wrap::cuda::malloc2DSurfaceObject(&variables, NELR, NVAR);
    initialize_variables(NELR, variables);

    wrap::cuda::SurfaceObject<float> old_variables;
    wrap::cuda::malloc2DSurfaceObject(&old_variables, NELR, NVAR);
    float *fluxes = alloc<float>(NELR * NVAR);
    float *step_factors = alloc<float>(NELR);

    // make sure all memory is doublely allocated before we start timing
    initialize_variables(NELR, old_variables);
    initialize_variables(NELR, fluxes);
    cudaMemset((void *)step_factors, 0, sizeof(float) * NELR);
    // make sure CUDA isn't still doing something before we start timing
    cudaThreadSynchronize();

    // these need to be computed the first time in order to compute time step
    std::cout << "Starting..." << std::endl;

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Begin iterations
    for (int i = 0; i < iterations; i++) {
        checkCudaErrors(cudaMemcpy2DArrayToArray(
            old_variables.devPtr, 0, 0,
            variables.devPtr, 0, 0, nbytes<float>(NELR), NVAR,
            cudaMemcpyDeviceToDevice));

        // for the first iteration we compute the time step
        compute_step_factor(NELR, variables, areas, step_factors);
        getLastCudaError("compute_step_factor failed");

        for (int j = 0; j < RK; j++) {
            compute_flux(NELR, elements_surrounding_elements, normals,
                         variables, fluxes);
            getLastCudaError("compute_flux failed");
            time_step(j, NELR, old_variables, variables, step_factors, fluxes);
            getLastCudaError("time_step failed");
        }
    }

    cudaThreadSynchronize();
    sdkStopTimer(&timer);

    std::cout << (sdkGetAverageTimerValue(&timer) / 1000.0) / iterations
              << " seconds per iteration" << std::endl;

    if (getenv("OUTPUT")) {
        std::cout << "Saving solution..." << std::endl;
        dump(variables, NEL, NELR);
        std::cout << "Saved solution..." << std::endl;
    }

    std::cout << "Cleaning up..." << std::endl;
    dealloc<float>(areas);
    dealloc<int>(elements_surrounding_elements);
    dealloc<float>(normals);

    wrap::cuda::freeSurfaceObject(&variables);
    wrap::cuda::freeSurfaceObject(&old_variables);
    dealloc<float>(fluxes);
    dealloc<float>(step_factors);

    std::cout << "Done..." << std::endl;

    return 0;
}

/* vim: set ts=4 sw=4 et cindent: */
