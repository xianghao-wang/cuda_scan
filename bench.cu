#include <cstdio>

#include "scanner.cuh"
#include "hs_scanner.cu"
#include "updown_scanner.cu"

int main(int argc, char **argv)
{
    int n, i;
    double *In, *Out;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: ./bench <size>\n");
        exit(1);
    }
    n = atoi(argv[1]);

    In = new double[n];
    Out = new double[n];
    for (i = 0; i < n; ++ i)
        In[i] = i;

    auto hs = new UpDownScanner<double>(In, Out, n);
    hs->begin();
    hs->run();
    hs->end();

    for (i = 0; i < n; ++ i)
        printf("%lf\n", Out[i]);

    return 0;
}