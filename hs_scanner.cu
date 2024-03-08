/**
 * Hillis and Steele scan
 * 
 * This is a naive approach implementing scan, which is not
 * work-efficient.
 * 
 * Work: O(nlogn)
 * Span: O(logn)
 * */

#include "scanner.cuh"

template <typename T>
class HSScanner: public Scanner<T> {
public:
    HSScanner(const T *In, T *Out, int n): Scanner<T>(In, Out, n) {};
    virtual void run();
};

template <typename T>
static __global__ void hs_scan(const T *in, T *out, int n)
{
    extern __shared__ T cached[];
    unsigned int tid, d;
    int pin, pout;

    tid = threadIdx.x;
    pout = 0, pin = 1;

    // Put all inputs to shared memory
    cached[pout * n + tid] = in[tid];
    __syncthreads();

    for (d = 1; d < n; d *= 2)
    {
        // Swap out and in buffer
        pout = 1 - pout;
        pin = 1 - pout;

        if (tid >= d)
            cached[pout * n + tid] = cached[pin * n + tid] + cached[pin * n + tid - d];
        else
            cached[pout * n + tid] = cached[pin * n + tid];

        __syncthreads();
    }

    out[tid] = cached[pout * n + tid];
}


template <typename T>
void HSScanner<T>::run() {
    int shared_size = this->n * sizeof(T);
    hs_scan<<<1, this->n, shared_size>>>(this->In_dev, this->Out_dev, this->n);
    cudaDeviceSynchronize();
}