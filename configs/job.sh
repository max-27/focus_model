#!/bin/bash
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err

module restore training_max
source activate focus
conda info
wandb agent djchewbacca/FocusModelResize/41537j33