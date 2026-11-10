#!/bin/bash
#SBATCH -J mmsa_test
#SBATCH -p dgx2 
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o %j.out
#SBATCH -e %j.err

module load miniconda3/24.3.0
source activate mmsa
export HF_ENDPOINT=https://hf-mirror.com
cd ~/mmsa_project
python run_mosi.py
