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
eval_batch_size=256
batch_size=256
num_layers=4
hidden_c=300
hidden_s=75
hidden_size=100
patch_size=8
wd=5e-5
lr=1e-3
denom=1
fd_degree=0
fd_lambda=0
dataset='c100'
model='effkan_mixer'
num_grids=25
checkpoint_epoch=0
epochs=600
cutmix_prob=0.5
mixup_prob=0
grid_min=-0.5
grid_max=0.5
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
python main.py --init 'default' --fd-degree $fd_degree --fd-lambda $fd_lambda --u-norm 0 --u-epoch 1 --w-norm 0 --init-scale 0.11 --lr $lr --weight-decay $wd --num-grids $num_grids --grid-type 'uniform' --denominator $denom --grid-min $grid_min --grid-max $grid_max --dataset $dataset --model $model --autoaugment --epochs $epochs --eval-batch-size $eval_batch_size --num-workers $num_workers --cutmix-prob $cutmix_prob --mixup-prob $mixup_prob --patch-size $patch_size --hidden-c $hidden_c --hidden-s $hidden_s --hidden-size $hidden_size  --batch-size $batch_size --num-layers $num_layers --skip-min 1.0 --checkpoint-epoch $checkpoint_epoch 
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


