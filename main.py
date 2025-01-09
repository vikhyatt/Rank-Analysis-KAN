import argparse

import torch


import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import wandb
import datetime



from dataloader import get_dataloaders
from utils import get_model
from train import Trainer
import shlex
#from metrics import generate_plots

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args, experiment_name):
    setup(rank, world_size)
    # Original Code
    args.rank = rank
    if rank == 0:
        wandb.init(
            project='mlp_mixer',
            config=args,
            name=experiment_name,
        )

    # Sync all processes before starting training
    #dist.barrier()
    
    #with wandb.init(project='mlp_mixer', config=args, name=experiment_name):
    train_dl, test_dl = get_dataloaders(args)
    model = get_model(args)
    model_configs = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}_{args.lr}_PS{args.patch_size}_HSize{args.hidden_size}_HC{args.hidden_c}_HS{args.hidden_s}_NL{args.num_layers}_SM{args.skip_min}_PARAMS{sum(p.numel() for p in model.parameters() if p.requires_grad)}_init{args.init}_scale_{args.init_scale}_NG{args.num_grids}_Grid{args.grid_type}{args.grid_min},{args.grid_max}_WD{args.weight_decay}_D{args.denominator}_WN{args.w_norm}_FD{args.fd_degree, args.fd_lambda},F{args.use_fourier}"
    args.model_configs = model_configs
    
    print('Number of Learnable Parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = Trainer(model, args)
    trainer.fit(train_dl, test_dl, init = args.init)
         
    #model = MyModel()
    
    # Compile the model (optional, for performance boost)
    #model = torch.compile(model)
    
    # Move model to the correct GPU
    #model.to(rank)
    
    # Wrap model in DDP
    #ddp_model = DDP(model, device_ids=[rank])
    
    # Your training code here
    #train_loop(ddp_model)
    
    cleanup()


def parse_exp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'imgnet','imgnet200','imgnet1k'])
    parser.add_argument('--model', required=True, choices=['mlp_mixer', 'kan_mixer', 'effkan_mixer','fasterkan_mixer','kat_mixer', 'hire_mixer', 'hire_mixer_2','hire_kan'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1028)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--checkpoint-epoch', type=int, default=0)
    parser.add_argument('--init', default='default', choices=['default', 'uniform', 'xavier', 'beta', 'gamma', 'exponential', 'zero', 'orthogonal'])
    parser.add_argument('--init-scale', nargs='+', type=float, default=[0.1])
    parser.add_argument('--grid-min', type=float, default=-2)
    parser.add_argument('--grid-max', type=float, default=2)
    parser.add_argument('--denominator', type=float, default=0)
    parser.add_argument('--grid-type', default='uniform', choices=['chebyshev', 'uniform'])
    parser.add_argument('--w-norm', type=float, default=0)
    parser.add_argument('--use-fourier', type=float, default=0)
    parser.add_argument('--u-norm', type=float, default=0)
    parser.add_argument('--u-epoch', type=int, default=0)
    parser.add_argument('--fd-degree', type=int, default=0)
    parser.add_argument('--fd-lambda', type=float, default=0.0)
    # parser.add_argument('--precision', type=int, default=16)
    
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--hidden-c', type=int, default=512)
    parser.add_argument('--hidden-s', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-grids', type=int, default=8)
    parser.add_argument('--drop-p', type=float, default=0.)
    parser.add_argument('--use-ffcv', type=float, default=0.)
    parser.add_argument('--off-act', action='store_true', help='Disable activation function')
    parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.')
    parser.add_argument('--skip-min', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'lion', 'LBFGS', 'adamw'])
    parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine','none'])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--off-nesterov', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--warmup-epoch', type=int, default=5)
    parser.add_argument('--autoaugment', action='store_true')
    parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
    parser.add_argument('--cutmix-beta', type=float, default=1.0)
    parser.add_argument('--cutmix-prob', type=float, default=0.)
    parser.add_argument('--mixup-prob', type=float, default=0.)
    parser.add_argument('--mixup-beta', type=float, default=1.0)
    parser.add_argument('--rand-augment-mag', type=int, default=15)
    parser.add_argument('--rand-augment-ops', type=int, default=2)
    parser.add_argument('--re-prob', type=float, default=0.1)
    
    
    args = parser.parse_args()
    #torch.cuda.set_device(1)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.grid = [args.grid_min, args.grid_max]
    print(f'Device using: {args.device}')
    args.nesterov = not args.off_nesterov
    torch.random.manual_seed(args.seed)
    
    experiment_name = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}"
    
    if args.autoaugment:
        experiment_name += "_aa"
    if args.clip_grad:
        experiment_name += f"_cg{args.clip_grad}"
    if args.off_act:
        experiment_name += f"_noact"
    if args.cutmix_prob>0.:
        experiment_name += f'_cm'
    if args.is_cls_token:
        experiment_name += f"_cls"

    return args, experiment_name





if __name__=='__main__':
    #world_size = 2
    wandb.login()
    args, experiment_name = parse_exp_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args, experiment_name,), nprocs=world_size)
        #print(args_list)
                
        #generate_plots(model, test_dl, experiment_name)
