#!/bin/sh

#SBATCH -o mylog.out-%j
#SBATCH -n 2
#SBATCH -N 2

# Initialize Modules
source /etc/profile

# Load Julia and MPI Modules
module load julia-latest
module load mpi/mpich-x86_64

~/.julia/bin/mpiexecjl -n 2 julia --project=. mpihelloworld.jl