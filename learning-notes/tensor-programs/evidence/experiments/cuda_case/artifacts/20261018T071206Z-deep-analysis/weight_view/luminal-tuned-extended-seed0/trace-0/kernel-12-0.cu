extern "C" __global__ void k(const __half* a, const __half* b, __half* out, const long long* params) {
    const unsigned long long n = 1048576LL;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long rem = i;
    long long c1 = (long long)(rem % 1024LL); rem /= 1024LL;
    long long c0 = (long long)(rem % 1024LL); rem /= 1024LL;
    long long b_idx = 0LL + c1;
    out[i] = a[i] + b[b_idx];
}