long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float (*p)[NCOLS],
                            cudaTextureObject_t tIn,
                            float (*tOut)[NCOLS], float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    float amb_temp = 80.0f;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int x = i;
    int y = j;

    int W = (x == 0) ? x : x - 1;
    int E = (x == nx - 1) ? x : x + 1;

    for (int k = 0; k < nz; ++k) {
        int row = k * ny + y; /* linearized z and y */
        int row_north = (y == 0) ? row : row - 1;
        int row_south = (y == ny - 1) ? row : row + 1;

        float center = wrap::ptx::tex2Ds32<float>(tIn, x, row);
        float left = wrap::ptx::tex2Ds32<float>(tIn, W, row);
        float right = wrap::ptx::tex2Ds32<float>(tIn, E, row);
        float north = wrap::ptx::tex2Ds32<float>(tIn, x, row_north);
        float south = wrap::ptx::tex2Ds32<float>(tIn, x, row_south);
        float bottom =
            (k == 0) ? center : wrap::ptx::tex2Ds32<float>(tIn, x, row - ny);
        float top =
            (k == nz - 1)
                ? center
                : wrap::ptx::tex2Ds32<float>(tIn, x, row + ny);

        tOut[row][x] = cc * center + cw * left + ce * right + cs * south +
                       cn * north + cb * bottom + ct * top + sdc * p[row][x] +
                       ct * amb_temp;
    }
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut, int nx, int ny, int nz,
                  float Cap, float Rx, float Ry, float Rz, float dt,
                  int numiter) {
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;

    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

    size_t s = sizeof(float) * nx * ny * nz;
    float *tOut_d, *p_d;
    wrap::cuda::TextureObject<float> tIn_d;
    cudaMalloc((void **)&p_d, s);
    cudaMalloc((void **)&tIn_d, s);
    wrap::cuda::malloc2DTextureObject(&tIn_d, nx, ny * nz);
    cudaMalloc((void **)&tOut_d, s);
    wrap::cuda::memcpy2DTextureObject(&tIn_d, tIn, nx, ny * nz);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);

    long long start = get_time();
    assert(numiter == 1);
    //for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>(
            (float (*)[NCOLS])p_d,
            tIn_d.tex,
            (float (*)[NCOLS])tOut_d,
            stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
        //float *t = tIn_d;
        //tIn_d = tOut_d;
        //tOut_d = t;
    //}
    cudaDeviceSynchronize();
    long long stop = get_time();
    float time = (float)((stop - start) / (1000.0 * 1000.0));
    printf("Time: %.3f (s)\n", time);
    cudaMemcpy(tOut, tOut_d, s, cudaMemcpyDeviceToHost);
    cudaFree(p_d);
    wrap::cuda::freeTextureObject(&tIn_d);
    cudaFree(tOut_d);
    return;
}
