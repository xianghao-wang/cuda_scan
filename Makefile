NVCC := nvcc

CFLAGS += -I.

EXES := bench

bench: bench.cu hs_scanner.cu scanner.cuh
	$(NVCC) $(CFLAGS) -o $@ $<