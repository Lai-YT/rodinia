long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p,
                            cudaTextureObject_t tIn,
                            float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0) ? c : c - 1;
    int E = (i == nx - 1) ? c : c + 1;
    int N = (j == 0) ? c : c - nx;
    int S = (j == ny - 1) ? c : c + nx;

    auto fetch = [&](int idx) {
        int tx = idx % nx;
        int ty = idx / nx;
        return wrap::ptx::tex2Ds32<float>(tIn, tx, ty);
    };

    float temp1, temp2, temp3;
    temp1 = temp2 = fetch(c);
    temp3 = fetch(c + xy);

    /* first layer (k = 0) */
    tOut[c] = cc * temp2 + cw * fetch(W) + ce * fetch(E) +
              cs * fetch(S) + cn * fetch(N) + cb * temp1 + ct * temp3 +
              sdc * p[c] + ct * amb_temp;

    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = fetch(c + xy);

        tOut[c] = cc * temp2 + cw * fetch(W) + ce * fetch(E) +
                  cs * fetch(S) + cn * fetch(N) + cb * temp1 + ct * temp3 +
                  sdc * p[c] + ct * amb_temp;

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }

    /* last layer (k = nz-1) */
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * fetch(W) + ce * fetch(E) +
              cs * fetch(S) + cn * fetch(N) + cb * temp1 + ct * temp3 +
              sdc * p[c] + ct * amb_temp;
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
    wrap::cuda::malloc2DTextureObject(&tIn_d, nx, ny * nz);
    cudaMalloc((void **)&tOut_d, s);
    wrap::cuda::memcpy2DTextureObject(&tIn_d, tIn, nx, ny * nz);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);

    long long start = get_time();
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>(p_d, tIn_d.tex, tOut_d, stepDivCap, nx,
                                             ny, nz, ce, cw, cn, cs, ct, cb,
                                             cc);
        // if (i != numiter - 1) {
        //     float *t = tIn_d;
        //     tIn_d = tOut_d;
        //     tOut_d = t;
        // }
    }
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
