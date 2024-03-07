#ifndef SCANNER_CUH
#define SCANNER_CUH

template <typename T>
class Scanner {
public:
    Scanner(const T *In, T *Out, int n): In(In), Out(Out), n(n) {}
    void begin();
    virtual void run() {};
    void end();

protected:
    int n;
    const T *In;
    T *Out, *In_dev, *Out_dev;
};

template <typename T>
static void scan_seq(const T *In, T *Out, int n)
{
    int i;

    Out[0] = In[0];
    for (i = 1; i < n; ++ i)
        Out[i] = In[i] + Out[i - 1];
}

template <typename T>
void Scanner<T>::begin() {
    cudaMalloc(&In_dev, n * sizeof(T));
    cudaMalloc(&Out_dev, n * sizeof(T));
    cudaMemcpy(In_dev, In, n * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void Scanner<T>::end() {
    cudaMemcpy(Out, Out_dev, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(In_dev);
    cudaFree(Out_dev);
}

#endif