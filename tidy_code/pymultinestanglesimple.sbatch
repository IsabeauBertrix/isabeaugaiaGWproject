#!/bin/bash
#SBATCH --job-name=PyMultiNestMPI1
#SBATCH --output=PyMultiNestMPI1.out
#SBATCH --error=PyMultiNestMPI1.err
#SBATCH --mail-user=isabeau.bertrix@sfr.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=72:00:00
#SBATCH --mem=12G

MPI_DIR=/usr/local/mpich

export LD_LIBRARY_PATH=/home/cjmoore/MyLibraries/lib/multinest/lib:$MPI_DIR/lib:$LD_LIBRARY_PATH
export PATH=$MPI_DIR/bin:$PATH

RUNPATH=/home/isabeau/isabeaugaiaGWproject/tidy_code/
cd $RUNPATH

export outputfiles_dir=$RUNPATH"/chains1-"$SLURM_JOB_ID

export OMP_NUM_THREADS=1

if [ ! -d $outputfiles_dir ]; then
  mkdir $outputfiles_dir
fi

source /home/isabeau/pymultinest_multi/venv/bin/activate

mpiexec -n 48 python gaia_with_multinest_simple.py

deactivate
