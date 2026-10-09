#!/bin/bash
#SBATCH --job-name=train-car
#SBATCH --output=/home/bcostanza/wf-curve-pixell/logs/train_%j.out
#SBATCH --partition=apollo
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00                # cluster limit (4320 min)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bcostanza@ias.edu

# Training of DeepWiener_threechannels with the curved-sky J3 loss (src/training_car.py).
# Submit from the repository root: sbatch slurm/train.sh
# A different config (e.g. a short test): WF_CONFIG=/path/config.dict sbatch slurm/train.sh
# 10 trials (~78 h) exceed the 3-day limit: split them, e.g.
#   N_TRIALS=5 sbatch slurm/train.sh ;  N_TRIALS=5 sbatch --dependency=afterany:<jobid> slurm/train.sh
# (same Optuna study, sqlite, load_if_exists)
# CPUs: sht_nthread (16) for the spherical-harmonic transforms + 4 DataLoader workers.

# ---- environment: edit to match your installation ---------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch_cmb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SLURM_SUBMIT_DIR}/src"

git -C "${SLURM_SUBMIT_DIR}" log -1 --format='commit %H'
echo "config: ${WF_CONFIG:-src/config.dict}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python -u training_car.py
