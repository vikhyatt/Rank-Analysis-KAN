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
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'lion', 'LBFGS'])
parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine','none'])
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--weight-decay', type=float, default=5e-5)
parser.add_argument('--off-nesterov', action='store_true')
#parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--warmup-epoch', type=int, default=5)
parser.add_argument('--autoaugment', action='store_true')
parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
parser.add_argument('--cutmix-beta', type=float, default=1.0)
parser.add_argument('--cutmix-prob', type=float, default=0.)


parser.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')



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
        model_configs = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}_{args.lr}_PS{args.patch_size}_HSize{args.hidden_size}_HC{args.hidden_c}_HS{args.hidden_s}_NL{args.num_layers}_SM{args.skip_min}_PARAMS{sum(p.numel() for p in model.parameters() if p.requires_grad)}_init{args.init}_scale_{args.init_scale}_NG{args.num_grids}_Grid{args.grid_type}{args.grid_min},{args.grid_max}_WD{args.weight_decay}_D{args.denominator}_WN{args.w_norm}_FD{args.fd_degree, args.fd_lambda},F{args.use_fourier}"
        args.model_configs = model_configs
        
        print('Number of Learnable Parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
        trainer = Trainer(model, args)
        trainer.fit(train_dl, test_dl, init = args.init)
        #print(args_list)
                
        #generate_plots(model, test_dl, experiment_name)
