#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

inline cudaError_t cudaCheck(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s at %s:%d\n",
                cudaGetErrorString(result), file, line);
        exit(EXIT_FAILURE);
    }
    return result;
}
#define CUDA_CALL(val) cudaCheck((val), __FILE__, __LINE__)

template <typename T> T **cudaAllocateZeroedMatrix(int rows, int cols) {
    T **matrix;
    CUDA_CALL(cudaMallocManaged(&matrix, rows * sizeof(T *)));
    for (int i = 0; i < rows; ++i) {
        CUDA_CALL(cudaMallocManaged(&matrix[i], cols * sizeof(T)));
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

template <typename T> void cudaFreeMatrix(T **matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        CUDA_CALL(cudaFree(matrix[i]));
    }
    CUDA_CALL(cudaFree(matrix));
}

const int NX = 41;
const int NY = 41;
const int NT = 500;
const int NIT = 50;
constexpr double dx = 2.0 / (NX - 1);
constexpr double dx2 = dx * dx;
constexpr double dy = 2.0 / (NY - 1);
constexpr double dy2 = dy * dy;
const double dt = 0.01;
const double rho = 1.0;
const double nu = 0.02;

__constant__ double dc_dx, dc_dy, dc_dt, dc_rho, dc_nu;

__device__ double compute_new_u(double **u, double **p, int j, int i) {
    double new_u = u[j][i];
    new_u -= u[j][i] * dc_dt / dc_dx * (u[j][i] - u[j][i - 1]);
    new_u -= u[j][i] * dc_dt / dc_dy * (u[j][i] - u[j - 1][i]);
    new_u -= dc_dt / (2 * dc_rho * dc_dx) * (p[j][i + 1] - p[j][i - 1]);
    new_u += dc_nu * dc_dt / dx2 * (u[j][i + 1] - 2 * u[j][i] + u[j][i - 1]);
    new_u += dc_nu * dc_dt / dy2 * (u[j + 1][i] - 2 * u[j][i] + u[j - 1][i]);
    return new_u;
}

__device__ double compute_new_v(double **v, double **p, int j, int i) {
    double new_v = v[j][i];
    new_v -= v[j][i] * dc_dt / dc_dx * (v[j][i] - v[j][i - 1]);
    new_v -= v[j][i] * dc_dt / dc_dy * (v[j][i] - v[j - 1][i]);
    new_v -= dc_dt / (2 * dc_rho * dc_dy) * (p[j + 1][i] - p[j - 1][i]);
    new_v += dc_nu * dc_dt / dx2 * (v[j][i + 1] - 2 * v[j][i] + v[j][i - 1]);
    new_v += dc_nu * dc_dt / dy2 * (v[j + 1][i] - 2 * v[j][i] + v[j - 1][i]);
    return new_v;
}

__device__ double compute_new_p(double **p, double **b, int j, int i) {
    double new_p = dy2 * (p[j][i + 1] + p[j][i - 1]);
    new_p += dx2 * (p[j + 1][i] + p[j - 1][i]);
    new_p -= b[j][i] * dx2 * dy2;
    new_p /= 2 * (dx2 + dy2);
    return new_p;
}

__device__ void set_boundary_conditions_u(double **u) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == NX + 1) {
        for (auto i = 0; i < NX; i++)
            u[0][i] = 0.0;
        for (auto j = 0; j < NY; j++)
            u[j][0] = 0.0;
        for (auto j = 0; j < NY; j++)
            u[j][NX - 1] = 0.0;
        for (auto i = 0; i < NX; i++)
            u[NY - 1][i] = 1.0;
    }
}

__device__ void set_boundary_conditions_v(double **v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == NX + 1) {
        for (auto i = 0; i < NX; i++)
            v[0][i] = 0.0;
        for (auto i = 0; i < NX; i++)
            v[NY - 1][i] = 0.0;
        for (auto j = 0; j < NY; j++)
            v[j][0] = 0.0;
        for (auto j = 0; j < NY; j++)
            v[j][NX - 1] = 0.0;
    }
}

__device__ void set_boundary_conditions_p(double **p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == NX + 1) {
        for (auto j = 0; j < NY; j++)
            p[j][NX - 1] = p[j][NX - 2];
        for (auto i = 0; i < NX; i++)
            p[0][i] = p[1][i];
        for (auto j = 0; j < NY; j++)
            p[j][0] = p[j][1];
        for (auto i = 0; i < NX; i++)
            p[NY - 1][i] = 0.0;
    }
}

__global__ void kernel(double **u, double **v, double **p, double **b) {
    bool active_thread = true;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NY * NX)
        active_thread = false;
    auto j = tid / NX;
    auto i = tid % NX;
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1)
        active_thread = false;
    auto grid = cooperative_groups::this_grid();

    if (active_thread) {
        b[j][i] = (u[j][i + 1] - u[j][i - 1]) / (2 * dc_dx) +
                  (v[j + 1][i] - v[j - 1][i]) / (2 * dc_dy);
        b[j][i] /= dc_dt;
        b[j][i] -= std::pow((u[j][i + 1] - u[j][i - 1]) / (2 * dc_dx), 2);
        b[j][i] -= 2 * (u[j + 1][i] - u[j - 1][i]) / (2 * dc_dy) *
                   (v[j][i + 1] - v[j][i - 1]) / (2 * dc_dx);
        b[j][i] -= std::pow((v[j + 1][i] - v[j - 1][i]) / (2 * dc_dy), 2);
        b[j][i] *= dc_rho;
    }
    grid.sync();

    for (auto it = 0; it < NIT; it++) {
        double new_p;
        if (active_thread) {
            new_p = compute_new_p(p, b, j, i);
        }
        grid.sync();
        if (active_thread)
            p[j][i] = new_p;

        set_boundary_conditions_p(p);
        grid.sync();
    }

    double new_u;
    if (active_thread) {
        new_u = compute_new_u(u, p, j, i);
    }
    grid.sync();
    if (active_thread)
        u[j][i] = new_u;
    grid.sync();

    double new_v;
    if (active_thread) {
        new_v = compute_new_v(v, p, j, i);
    }
    grid.sync();
    if (active_thread)
        v[j][i] = new_v;
    grid.sync();

    set_boundary_conditions_u(u);
    set_boundary_conditions_v(v);
}

int main() {
    double **u = cudaAllocateZeroedMatrix<double>(NY, NX);
    double **v = cudaAllocateZeroedMatrix<double>(NY, NX);
    double **p = cudaAllocateZeroedMatrix<double>(NY, NX);
    double **b = cudaAllocateZeroedMatrix<double>(NY, NX);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));

    CUDA_CALL(cudaMemcpyToSymbol(dc_dx, &dx, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dc_dy, &dy, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dc_dt, &dt, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dc_rho, &rho, sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(dc_nu, &nu, sizeof(double)));

    const auto N = NX * NY;
    const auto tpb = deviceProp.maxThreadsPerBlock;
    const auto num_blocks = (N + tpb - 1) / tpb;
    void *args[] = {(void *)&u, (void *)&v, (void *)&p, (void *)&b};

    std::ofstream ufile("u.dat"), vfile("v.dat"), pfile("p.dat");
    for (auto n = 0; n < NT; n++) {
        CUDA_CALL(
            cudaLaunchCooperativeKernel((void *)kernel, num_blocks, tpb, args));
        CUDA_CALL(cudaDeviceSynchronize());

        if (n % 10 == 0) {
            for (auto j = 0; j < NY; j++) {
                for (auto i = 0; i < NX; i++) {
                    ufile << u[j][i] << " ";
                    vfile << v[j][i] << " ";
                    pfile << p[j][i] << " ";
                }
            }
            ufile << "\n";
            vfile << "\n";
            pfile << "\n";
        }
    }

    cudaFreeMatrix(u, NY);
    cudaFreeMatrix(v, NY);
    cudaFreeMatrix(p, NY);
    cudaFreeMatrix(b, NY);

    return 0;
}
