#!/bin/bash
#SBATCH -J mmsa_baselines
#SBATCH -p dgx2
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Submit from your home folder (Slurm writes logs relative to where you submit):
#   cd ~ && sbatch multimodal-ml/experiments/mmsa_baselines/slurm_baselines.sh --dataset mosi
# Any arguments after the script name are passed to run_baselines.py.

module load miniconda3/24.3.0
source activate mmsa

# GPU nodes may have no internet: use BERT from the local cache only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/multimodal-ml/experiments/mmsa_baselines
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python run_baselines.py "$@"
