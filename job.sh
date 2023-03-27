#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-08:00
#SBATCH -p gpu_yu
#SBATCH --account=yu_ky98_contrib
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err

module load gcc/6.2.0
module load cuda/10.1
module load miniconda3/4.10.3
source activate focus
#wandb agent djchewbacca/SweepYNetSpectral/2r7ic6h9
#python src/eval.py
python src/train.py experiment=example #--config-name run_again.yaml