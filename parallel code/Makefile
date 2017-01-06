CUDAPATH = /Developer/NVIDIA/CUDA-7.5

NVCC = $(CUDAPATH)/bin/nvcc

# NVCCFLAGS = -I$(CUDAPATH)/include -O3 --compiler-options='-Wno-gnu-designator'
NVCCFLAGS = -I$(CUDAPATH)/include -g -G --compiler-options='-Wno-gnu-designator'

# LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

CAIRO_L := -L/usr/local/lib/cairo/ -lcairo 
CAIRO_I := -I/usr/local/include/cairo/ 


################################################################################

# SAMPLE_ENABLED := 1

# Gencode arguments
SMS ?= 20 30 35 37 50 52

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

# ifeq ($(SAMPLE_ENABLED),0)
# EXEC ?= @echo "[@]"
# endif

################################################################################

.SUFFIXES : .cu .ptx

BINARIES = main


main: main.o
	$(NVCC) $(GENCODE) $(GENCODE_FLAGS) $(CAIRO_L) $(NVCCFLAGS) -o $@ $< 

main.o:main.cu
	$(NVCC) $(CAIRO_I) $(GENCODE) $(GENCODE_FLAGS) $(NVCCFLAGS) -o $@ -c $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS)  -o $@ -c $<

run: build
	$(EXEC) ./main

clean:	
	rm -f main main.o
	rm -f *.o $(BINARIES)

clobber: clean
