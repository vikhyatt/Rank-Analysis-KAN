#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=80G           # Memory request for the job
#SBATCH --partition=h100     # Specify the partition
#SBATCH --time=4:00:00     # Set a time limit

# Activate your virtual environment
source ../venvs/kan/bin/activate

# Now, you can run your Python script
#python main.py --init 'uniform' --init-scale 0.02 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0

#python main.py --init 'uniform' --init-scale 0.002 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'uniform' --init-scale 0.0002 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'xavier' --init-scale 1.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'xavier' --init-scale 0.001 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

python main.py --init 'default' --init-scale 0.1 --num-grids 4 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

python main.py --init 'default' --init-scale 0.1 --num-grids 2 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

python main.py --init 'uniform' --init-scale 0.02 --num-grids 4 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

python main.py --init 'uniform' --init-scale 0.02 --num-grids 2 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 


