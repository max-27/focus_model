#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-10:00
#SBATCH -p gpu_yu
#SBATCH --account=yu_ky98_contrib
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err

module load gcc/6.2.0
module load cuda/10.1
module load miniconda3/4.10.3
source activate focus
#wandb agent djchewbacca/SweepYNetMixed/uv1im5cb
#python src/eval.py
python src/train.py experiment=example #--config-name run_again.yaml