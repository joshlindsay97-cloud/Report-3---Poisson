#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --job-name=ph510_task2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load mpi/latest
source .venv/bin/activate

python -c "from mpi4py import MPI; print('MPI:', MPI.Get_library_version().splitlines()[0])"

mkdir -p out
mpirun -np $SLURM_NTASKS python walk.py --n 51 --length 1.0 --start-x 0.5 --start-y 0.5 --walkers 200000 --out out/greens.npz
