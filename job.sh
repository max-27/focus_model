#!/bin/bash
#SBATCH -c 8
#SBATCH -t 0-01:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err

module load gcc/6.2.0
module load cuda/10.1
module load miniconda3/4.10.3
source activate focus
#wandb agent djchewbacca/SweepYNetJiang/zsq52qr4
python src/eval.py
