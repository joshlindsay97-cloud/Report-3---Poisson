#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --job-name=ph510_task1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

python --version
python run.py --n 51 --length 1.0 --bc all100 --charge none


set -euo pipefail


# Use a venv in your project folder (recommended)
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip >/dev/null

# numpy is required by the solver
pip install numpy >/dev/null

# Run Task 1 demo
python run.py --n 51 --length 1.0 --bc all100 --charge none --tol 1e-8 --max-iters 200000
