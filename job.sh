#!/bin/bash
#SBATCH -c 8
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:4
#SBATCH --mem=8G
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err

module load gcc/6.2.0
module load cuda/10.1
module load miniconda3/4.10.3
source activate focus
wandb agent djchewbacca/SweepYNetJiang/bm3raoht
