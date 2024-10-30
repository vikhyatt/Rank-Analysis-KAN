#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2        # Request 1 GPU
#SBATCH --mem=120G           # Memory request for the job
#SBATCH --time=4:00:00     # Set a time limit

# Activate your virtual environment
source ../venvs/kan/bin/activate
num_workers=12
eval_batch_size=128
batch_size=128
num_layers=9
hidden_c=900
hidden_s=112
hidden_size=225
patch_size=16

# Now, you can run your Python script
#python main.py --init 'uniform' --init-scale 0.02 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0

#python main.py --init 'uniform' --init-scale 0.002 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'uniform' --init-scale 0.0002 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'xavier' --init-scale 1.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'xavier' --init-scale 0.001 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'default' --init-scale 0.11 --weight-decay 0 --num-grids 8 --grid-type 'uniform' --denominator 1.0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'default' --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'chebyshev' --denominator 0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'default' --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'chebyshev' --denominator 1.0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'default' --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'uniform' --denominator -2.0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
python main.py --init 'default' --w-norm 1 --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'uniform' --denominator -2.0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size $patch_size --hidden-c $hidden_c --hidden-s $hidden_s --hidden-size $hidden_size  --batch-size $batch_size --num-layers $num_layers --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'zero' --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'uniform' --denominator 0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'zero' --init-scale 0.11  --weight-decay 5e-5 --num-grids 8 --grid-type 'uniform' --denominator 0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'zero' --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'chebyshev' --denominator 1.0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'zero' --init-scale 0.11  --weight-decay 0 --num-grids 8 --grid-type 'chebyshev' --denominator 1.0 --grid-min -2.0 --grid-max 2.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'beta' --init-scale 5.0 5.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'beta' --init-scale 1.0 1.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'beta' --init-scale 10.0 10.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'beta' --init-scale 2.0 5.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'beta' --init-scale 10.0 3.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'gamma' --init-scale 10.0 1.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'gamma' --init-scale 10.0 5.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'gamma' --init-scale 1.0 5.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'gamma' --init-scale 1.0 1.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'exponential' --init-scale 3.5 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'exponential' --init-scale 7.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'exponential' --init-scale 1.0 --num-grids 8 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size $batch_size --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'uniform' --init-scale 0.01 --num-grids 8 --grid-min -1.0 --grid-max 1.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 
#python main.py --init 'xavier' --init-scale 0.1 --num-grids 8 --grid-min -1.0 --grid-max 1.0 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'default' --init-scale 0.1 --num-grids 2 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'uniform' --init-scale 0.02 --num-grids 4 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 

#python main.py --init 'uniform' --init-scale 0.02 --num-grids 2 --dataset imgnet --model kan_mixer --autoaugment --epochs 600 --eval-batch-size 128 --num-workers 12 --cutmix-prob 0.5 --patch-size 8 --hidden-c 512 --hidden-s 64 --hidden-size 128  --batch-size 256 --num-layers 4 --skip-min 1.0 --checkpoint-epoch 0 


