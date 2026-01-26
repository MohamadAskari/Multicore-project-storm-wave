# --- Execution Parameters ---
export OMP_NUM_THREADS=32
MPI_PROCS=4

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) \
               --bind-to none

# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
MPI_OMP_EXTRA_CFLAGS = -I/opt/homebrew/Cellar/open-mpi/5.0.9/include -I/opt/homebrew/opt/libomp/include
MPI_OMP_EXTRA_LIBS = -L/opt/homebrew/Cellar/open-mpi/5.0.9/lib -L/opt/homebrew/opt/libomp/lib -lomp

# Override the OMPFLAG after Makefile sets it (using override directive)
override OMPFLAG=-Xclang -fopenmp

# Flags for CUDA code
# Uncomment and add extra flags if you need them
#CUDA_EXTRA_CFLAGS =
#CUDA_EXTRA_LIBS =