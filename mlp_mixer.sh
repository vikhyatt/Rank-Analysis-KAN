#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=40G           # Memory request for the job
#SBATCH --partition=h100     # Specify the partition
#SBATCH --time=12:00:00     # Set a time limit

# Activate your virtual environment
source ../venvs/kan/bin/activate

# Now, you can run your Python script
python main.py --dataset imgnet --model mlp_mixer --autoaugment --epochs 600 --eval-batch-size 256 --num-workers 12 --cutmix-prob 0.5 --patch-size 16 --hidden-c 2048 --hidden-s 256 --hidden-size 512  --batch-size 256 --num-layers 8 --skip-min 1.0 --checkpoint-epoch 20
