import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


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
            use_same_fn = True,
            use_hankel = False,
            use_same_weight = False,
            use_pe = False,
            use_cpd = False,
            use_softmax_prod = False,
            num_grids = args.num_grids,
            skip_min = args.skip_min,
            init = args.init,
            spline_weight_init_scale = args.init_scale,
            grid = args.grid,
            grid_type = args.grid_type,
            denominator = args.denominator,
            w_norm = args.w_norm, 
            use_fourier = args.use_fourier
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
            use_same_fn = True,
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
    elif args.model=='hire_mixer':
        from hire_mixer import HireMLPNet
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        def hire_mlp_tiny(pretrained=False, **kwargs):
            layers = [2, 2, 4, 2]
            #layers = [2,4,2]
            mlp_ratios = [4, 4, 4, 4]
            #mlp_ratios = [4,4,4]
            embed_dims = [64, 128, 320, 512]
            #embed_dims = [64,128, 320]
            pixel = [4, 3, 3, 2]
            step_stride = [2, 2, 3, 2]
            step_dilation = [2, 2, 1, 1]
            step_pad_mode = 'c'
            pixel_pad_mode = 'c'
            model = HireMLPNet(
                layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
                step_stride=step_stride, step_dilation=step_dilation,
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, **kwargs)
            model.default_cfg = {
                    'url': '',
                    'num_classes': 100, 'input_size': (3, 224, 224), 'pool_size': None,
                    'crop_pct': 0.9, 'interpolation': 'bicubic',
                    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
                    **kwargs
                    }
            return model
        
        model =  hire_mlp_tiny()
    elif args.model=='hire_mixer_2':
        from hire_mixer_2 import HireMLPNet
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        def hire_mlp_tiny(size,classes, pretrained=False, **kwargs):
            layers = [2, 4,2]
            mlp_ratios = [4, 4, 4]
            embed_dims = [64, 128, 320]
            pixel = [4, 3, 3, 2]
            step_stride = [2, 2, 3, 2]
            step_dilation = [2, 2, 1, 1]
            step_pad_mode = 'c'
            pixel_pad_mode = 'c'
            model = HireMLPNet(
                layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
                step_stride=step_stride, step_dilation=step_dilation,
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, **kwargs)
            model.default_cfg = {
                    'url': '',
                    'num_classes': classes, 'input_size': (3, size, size), 'pool_size': None,
                    'crop_pct': 0.9, 'interpolation': 'bicubic',
                    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
                    **kwargs
                    }
            return model

        model =  hire_mlp_tiny(args.size, args.num_classes)

    elif args.model=='hire_kan':
        from hire_kan import HireKANNet
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        def hire_kan_tiny(size,classes,grids,use_same_fn, pretrained=False, **kwargs):
            layers = [2,2]
            mlp_ratios = [2, 2]
            embed_dims = [250, 200]
            pixel = [3,3]
            step_stride = [2, 2,]
            step_dilation = [2, 1]
            step_pad_mode = 'c'
            pixel_pad_mode = 'c'
            model = HireKANNet(
                layers, embed_dims=embed_dims, patch_size=7, mlp_ratios=mlp_ratios, pixel=pixel,
                step_stride=step_stride, step_dilation=step_dilation,num_classes = classes,
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, num_grids = grids, use_same_fn = use_same_fn, **kwargs)
            model.default_cfg = {
                    'url': '',
                    'num_classes': classes, 'input_size': (3, size, size), 'pool_size': None,
                    'crop_pct': 0.9, 'interpolation': 'bicubic',
                    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
                    **kwargs
                    }
            return model

        model =  hire_kan_tiny(args.size, args.num_classes, args.num_grids, use_same_fn = True)

    else:
        raise ValueError(f"No such model: {args.model}")
        
    if True:#torch.cuda.device_count() > 1:
        #torch.cuda.set_device(1)
        #model = nn.DataParallel(model, device_ids=[1,2,3]) 
        #model = torch.compile(model)
        #model.to(args.rank)
        #ddp_model = DDP(model, device_ids=[args.rank])
        model.to(args.rank)
        ddp_model = DDP(model, device_ids=[args.rank], find_unused_parameters=False)
        ddp_model = torch.compile(ddp_model, dynamic=True)
        
    else:
        print("Using Torch compile to speed up training")
        model = torch.compile(model) #Compile does not work with DataParallel
    
    #return model.to(args.device)
    return ddp_model

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
