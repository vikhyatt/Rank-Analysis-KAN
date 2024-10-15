import argparse

import torch
import wandb
wandb.login()

from dataloader import get_dataloaders
from utils import get_model
from train import Trainer
import shlex
#from metrics import generate_plots

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'imgnet'])
parser.add_argument('--model', required=True, choices=['mlp_mixer', 'kan_mixer', 'effkan_mixer','fasterkan_mixer','kat_mixer'])
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--eval-batch-size', type=int, default=1024)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--checkpoint-epoch', type=int, default=0)
parser.add_argument('--init', default='default', choices=['default', 'uniform', 'xavier', 'beta', 'gamma', 'exponential', 'zero'])
parser.add_argument('--init-scale', nargs='+', type=float, default=[0.1])
parser.add_argument('--grid-min', type=float, default=-2)
parser.add_argument('--grid-max', type=float, default=2)
parser.add_argument('--denominator', type=float, default=0)
parser.add_argument('--grid-type', default='uniform', choices=['chebyshev', 'uniform'])
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
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'lion', 'LBFGS'])
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

args = parser.parse_args()
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


if __name__=='__main__':
    with wandb.init(project='mlp_mixer', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        model = get_model(args)
        model_configs = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}_{args.lr}_PS{args.patch_size}_HSize{args.hidden_size}_HC{args.hidden_c}_HS{args.hidden_s}_NL{args.num_layers}_SM{args.skip_min}_PARAMS{sum(p.numel() for p in model.parameters() if p.requires_grad)}_init{args.init}_scale_{args.init_scale}_NG{args.num_grids}_Grid{args.grid_type}{args.grid_min},{args.grid_max}_WD{args.weight_decay}_D{args.denominator}"
        args.model_configs = model_configs
        
        print('Number of Learnable Parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
        trainer = Trainer(model, args)
        trainer.fit(train_dl, test_dl, init = args.init)
        #print(args_list)
                
        #generate_plots(model, test_dl, experiment_name)
