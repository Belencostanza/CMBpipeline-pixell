#!/bin/bash
#SBATCH --job-name=getbest-car
#SBATCH --output=/home/bcostanza/wf-curve-pixell/logs/getbest_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Prediction of the best Optuna trial on the first 10 validation maps (src/getbestmodel_car.py), on CPU.
# Submit from the repository root: sbatch slurm/getbest.sh

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch_cmb
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}/src"
git -C "${SLURM_SUBMIT_DIR}" log -1 --format='commit %H'
srun python -u getbestmodel_car.py
