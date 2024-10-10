import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

def get_model(args):
    model = None
    if args.model=='mlp_mixer':
        from mlp_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )
    elif args.model=='kan_mixer':
        from kan_mixer import KANMixer
        model = KANMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token,
            use_poly = False, 
            degree_poly = 2,
            use_base_update = True,
            base_activation = F.silu,
            use_same_fn = False,
            use_same_weight = False,
            use_pe = False,
            use_cpd = False,
            use_softmax_prod = False,
            num_grids = args.num_grids,
            skip_min = args.skip_min,
            init = args.init,
            spline_weight_init_scale = args.init_scale,
        )

    elif args.model=='kat_mixer':
        from kat_mixer import KATMixer
        model = KATMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token,
            use_poly = False, 
            degree_poly = 2,
            use_base_update = True,
            base_activation = F.silu,
            use_same_fn = False,
            use_same_weight = False,
            use_pe = False,
            use_cpd = False,
            use_softmax_prod = False,
            num_grids = args.num_grids,
            skip_min = args.skip_min,
        )

    elif args.model=='fasterkan_mixer':
        from fasterkan_mixer import FasterKANMixer
        model = FasterKANMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token,
            use_poly = False, 
            degree_poly = 2,
            use_base_update = True,
            base_activation = F.silu,
            use_same_fn = False,
            use_same_weight = False,
            use_pe = False,
            use_cpd = False,
            use_softmax_prod = False,
        )
        
    elif args.model=='effkan_mixer':
        from effkan_mixer import KANMixer
        model = KANMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token,
            base_activation = F.silu,
            enable_standalone_scale_spline = False,
            use_pe = True,
        )
    else:
        raise ValueError(f"No such model: {args.model}")
        
    return model.to(args.device)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
