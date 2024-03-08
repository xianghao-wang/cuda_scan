/**
 * Work-efficient scan including two phases - up & down
 * Work: O(n)
 * Span: O(logn)
 * */

#include "scanner.cuh"

template <typename T>
class UpDownScanner: public Scanner<T> {
public:
    UpDownScanner(const T *In, T *Out, int n): Scanner<T>(In, Out, n) {};
    virtual void run();
};

template <typename T>
static __global__ void ud_scan(const T *in, T *out, int n)
{
    extern __shared__ T cached[];
    unsigned int tid, d, offset, padded_n, idx;

    tid = threadIdx.x;
    padded_n = blockDim.x * 2;

    /* Up stage */
    // Move input array to shared memory
    cached[2 * tid] = 2 * tid < n ? in[2 * tid] : 0;
    cached[2 * tid + 1] = 2 * tid + 1 < n ? in[2 * tid + 1] : 0;

    // Implicitly construct a full binary tree
    offset = 1;
    for (d = padded_n >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            idx = 2 * offset * (tid + 1) - 1; // right child index
            cached[idx] += cached[idx - offset];
        }

        __syncthreads();
        offset *= 2;
    }

    /* Down stage */
    // @xianghao: Down stage looks like binary indexed tree
    // TODO: finish down stage


    if (2 * tid < n)
        out[2 * tid] = cached[2 * tid];
    if (2 * tid + 1 < n)
        out[2 * tid + 1] = cached[2 * tid + 1];
}

static unsigned int padding(unsigned int n)
{
    n --;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n ++;

    return n;
}

template <typename T>
void UpDownScanner<T>::run() {
    unsigned int padded_n, shared_size;

    // Padding n to the power of 2
    padded_n = padding(this->n);

    shared_size = padded_n * sizeof(T);

    ud_scan<<<1, padded_n / 2, shared_size>>>(this->In_dev, this->Out_dev, this->n);
}