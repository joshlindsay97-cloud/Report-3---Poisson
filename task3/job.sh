#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --job-name=ph510_task3
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load mpi/latest

# Pick ONE that exists:
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "../task2/.venv/bin/activate" ]; then
  source ../task2/.venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
  source ../.venv/bin/activate
else
  echo "Could not find a venv to activate." >&2
  exit 1
fi

# Optional: silence libfabric warning spam
export FI_PSM3_UUID=$(uuidgen)

# MPI smoke test (should print different ranks)
mpiexec -n $SLURM_NTASKS python -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank(), 'of', MPI.COMM_WORLD.Get_size())"

mkdir -p out

# Run the 3 Task 3 points
mpiexec -n $SLURM_NTASKS python walk_centre.py --walkers 200000 --out out/greens_center.npz
mpiexec -n $SLURM_NTASKS python walk_corner.py --walkers 200000 --out out/greens_corner.npz
mpiexec -n $SLURM_NTASKS python walk_face.py   --walkers 200000 --out out/greens_face.npz
