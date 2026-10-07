#!/bin/bash
#SBATCH --job-name=data-pixell
#SBATCH --output=/home/bcostanza/wf-curve-pixell/logs/data_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bcostanza@ias.edu

# Task 2 dataset: 1000 train + 100 valid CAR maps (src/config.dict).
# Submit from the repository root: sbatch slurm/data.sh
# (memory: X_train is 5.7 GB in float32; ~0.3 s per map on one core locally)

# ---- environment: edit to match your installation ---------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch_cmb
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# The scripts read src/config.dict (or the file given in $WF_CONFIG).
cd "${SLURM_SUBMIT_DIR}/src"

git -C "${SLURM_SUBMIT_DIR}" log -1 --format='commit %H'
srun python run_dataset.py
