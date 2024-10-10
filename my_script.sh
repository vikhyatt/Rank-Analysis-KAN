#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=40G           # Memory request for the job
#SBATCH --partition=h100     # Specify the partition
#SBATCH --time=12:00:00     # Set a time limit

# Activate your virtual environment
source ../venvs/kan/bin/activate

# Now, you can run your Python script
python main.py --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 256 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0
