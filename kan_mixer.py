import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchsummary
from fastkan import FastKAN as KAN
from fastkan import FastKANLayer as KANLinear

## Modified from CycleMLP
## Author: Jianyuan Guo (jyguo@pku.edu.cn)
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torch.nn.modules.utils import _pair

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

def get_sinusoid_encoding(num_tokens, token_len):
    """ Make Sinusoid Encoding Table

        Args:
            num_tokens (int): number of tokens
            token_len (int): length of a token
            
        Returns:
            (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HireMLP(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2,
                 step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        """
        self.pixel: h and w in inner-region rearrangement
        self.step: s in cross-region rearrangement
        """
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        #print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
        #      pixel, pixel_pad_mode, step, step_pad_mode))
        
        self.mlp_h1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim//2)
        self.mlp_h2 = nn.Conv2d(dim//2, dim*pixel, 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim//2)
        self.mlp_w2 = nn.Conv2d(dim//2, dim*pixel, 1, bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)
        
        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape

        pad_h, pad_w = (self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.clone(), x.clone()
        
        if self.step:
            if self.step_pad_mode == '0':
                h = F.pad(h, (0, 0, self.step, 0), "constant", 0)
                w = F.pad(w, (self.step, 0, 0, 0), "constant", 0)
                h = torch.narrow(h, 2, 0, H)
                w = torch.narrow(w, 3, 0, W)
            elif self.step_pad_mode == 'c':
                h = torch.roll(h, self.step, -2)
                w = torch.roll(w, self.step, -1)
                # h = F.pad(h, (0, 0, self.step, 0), mode='circular')
                # w = F.pad(w, (self.step, 0, 0, 0), mode='circular')
            else:
                raise NotImplementedError("Invalid pad mode.")

        if self.pixel_pad_mode == '0':
            h = F.pad(h, (0, 0, 0, pad_h), "constant", 0)
            w = F.pad(w, (0, pad_w, 0, 0), "constant", 0)
        elif self.pixel_pad_mode == 'c':
            h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
            w = F.pad(w, (0, pad_w, 0, 0), mode='circular')
        elif self.pixel_pad_mode == 'replicate':
            h = F.pad(h, (0, 0, 0, pad_h), mode='replicate')
            w = F.pad(w, (0, pad_w, 0, 0), mode='replicate')
        else:
            raise NotImplementedError("Invalid pad mode.")
        
        h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C*self.pixel, (H + pad_h) // self.pixel, W)
        w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C*self.pixel, H, (W + pad_w) // self.pixel)
          
        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)
        
        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)
        
        h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 4, 2).reshape(B, C, H, W + pad_w)
        
        h = torch.narrow(h, 2, 0, H)
        w = torch.narrow(w, 3, 0, W)

        # cross-region arrangement operation
        if self.step and self.step_pad_mode == 'c':
            h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)
            # h = F.pad(h, (0, 0, 0, self.step), mode='circular')
            # w = F.pad(w, (0, self.step, 0, 0), mode='circular')
            # h = torch.narrow(h, 2, self.step, H)
            # w = torch.narrow(w, 3, self.step, W)

        c = self.mlp_c(x)

        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0] + w * a[1] + c * a[2]
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HireBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = HireMLP(
            dim, attn_drop=attn_drop, pixel=pixel, step=step,
            step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        print(f"Before Patch Embeddings: {x.shape}")
        x = self.proj(x)
        print(f"After Patch Embeddings: {x.shape}")
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """ Downsample stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = nn.BatchNorm2d(out_embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        x = self.norm(x)
        x = self.act(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=4., attn_drop=0., drop_path_rate=0.,
                 pixel=2, step_stride=2, step_dilation=1,
                 step_pad_mode='c', pixel_pad_mode='c', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            HireBlock(
                dim, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=block_dpr, pixel=pixel,
                step=(block_idx % step_stride) * step_dilation,
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode))
    blocks = nn.Sequential(*blocks)

    return blocks


class HireMLPNet(nn.Module):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, mlp_ratios=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        pixel=[2,2,2,2], step_stride=[2,2,2,2],step_dilation=[1,1,1,1],
        step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        print('drop path: {}'.format(drop_path_rate))

        self.num_classes = num_classes
        
        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i],
                attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, pixel=pixel[i],
                step_stride=step_stride[i], step_dilation=step_dilation[i],
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i+1], 2))

        self.network = nn.ModuleList(network)

        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(x.flatten(2).mean(2))
        return cls_out



class KANMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False, use_poly = False, degree_poly = 3, use_base_update = True, base_activation = F.silu, use_same_fn = False, use_same_weight = False, use_pe = False, use_cpd = False, use_softmax_prod = False, num_grids = 8, skip_min = 1, init = 'default',spline_weight_init_scale = 0.1, grid = [-1,-1]):
        super(KANMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        print(f"Polynomial Basis: {use_poly}, Degree of Polynomial: {degree_poly}, Using Same Function: {use_same_fn}, Using same weights: {use_same_weight}, Positional Embeddings: {use_pe}, CPD Decomposition: {use_cpd}, Softmax Prod: {use_softmax_prod}, Init: {init}")

        self.patch_emb = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            KANLinear((patch_size ** 2) * in_channels, hidden_size, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, init = init, spline_weight_init_scale = spline_weight_init_scale, grid_min = grid[0], grid_max = grid[1])
        )

        self.use_pe = use_pe
        if use_pe:
            self.pe = get_sinusoid_encoding(num_patches, hidden_size)

        #MLPs inserted  
        #self.MLPs = nn.Sequential(
        #    *[
        #        nn.Linear(hidden_size, hidden_size) 
        #    for _ in range(10)
        #    ]
        #)

        # HIRE-MLP
        #self.r1 = Rearrange('b p (h8 h16) -> b p h8 h16', h8=hidden_size // 8, h16=hidden_size // 16)
        #self.hireblocks = nn.Sequential(
       #     *[
       #     HireBlock(num_patches)
       #     for _ in range(50)
       #     ]
        #)
        #self.r2 = Rearrange('b p h8 h16 -> b p (h8 h16)')


        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        #MLP-Mixer blocks inserted  
        #self.mlp_mixer_layers = nn.Sequential(
        #    *[
        #        MLP_MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act) 
        #    for _ in range(2)
        #    ]
        #)
        delta_skip = (1-skip_min)/num_layers
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod,num_grids = num_grids, skip_param = (1 - delta_skip*i), init = init, spline_weight_init_scale = spline_weight_init_scale, grid= grid) 
            for i in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.clf = KANLinear(hidden_size, num_classes, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, init = init, spline_weight_init_scale = spline_weight_init_scale, grid_min = grid[0], grid_max = grid[1])


    def forward(self, x):
        out = self.patch_emb(x)

        #Use Positional Embeddings
        if self.use_pe:
            if self.pe.device != out.device:
                self.pe = self.pe.to(out.device)
                
            out = out + self.pe
            
        #HireMLP implementation
        #out = self.r1(out)
        #out = self.hireblocks(out)
        #out = self.r2(out)

        #Simple MLPs
        #out = self.MLPs(out)
        
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        
        #out = self.mlp_mixer_layers(out)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        return out



class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, use_poly = False, degree_poly = 3, use_base_update = True, base_activation = F.silu, use_same_fn = False, use_same_weight = False, use_cpd = False, use_softmax_prod = False, num_grids = 8, skip_param = 1, init = 'default', spline_weight_init_scale = 0.1, grid = [-1,1]):
        super(MixerLayer, self).__init__()
        self.kan1 = KAN1(num_patches, hidden_s, hidden_size, drop_p, off_act, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod,num_grids = num_grids, skip_param = skip_param, init = init, spline_weight_init_scale = spline_weight_init_scale, grid = grid)
        self.kan2 = KAN2(hidden_size, hidden_c, drop_p, off_act, use_poly = use_poly, degree_poly = degree_poly, use_base_update= use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, skip_param = skip_param, init = init, spline_weight_init_scale = spline_weight_init_scale, grid = grid)
    def forward(self, x):
        out = self.kan1(x)
        out = self.kan2(out)
        return out

class KAN1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act, use_poly = False, degree_poly = 3,
                  use_base_update = True, base_activation = F.silu, use_same_fn = False, use_same_weight = False, use_cpd = False, use_softmax_prod = False, num_grids = 8, skip_param = 1, init = 'default',spline_weight_init_scale = 0.1, grid = [-1,1]):
        super(KAN1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.skip_param = skip_param
        
        #self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.fc1 = KANLinear(num_patches, hidden_s, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, init = init,spline_weight_init_scale = spline_weight_init_scale, grid_min = grid[0], grid_max = grid[1])
        self.do1 = nn.Dropout(p=drop_p)
        #self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.fc2 = KANLinear(hidden_s, num_patches, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, init = init,spline_weight_init_scale = spline_weight_init_scale, grid_min = grid[0], grid_max = grid[1])
        
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        #out = self.do1(self.act(self.fc1(self.ln(x))))
        #out = self.do2(self.fc2(out))
        
        #initial_x = x.clone()
        #x = self.ln(x).permute(0,2,1)
        #x = self.fc1(x).permute(0,2,1)
        #out = self.do1(self.act(x)).permute(0,2,1)
        #out = self.fc2(out).permute(0,2,1)
        #out = self.do2(out)

        initial_x = x.clone()
        x = x.permute(0,2,1)
        x = self.fc1(x).permute(0,2,1)
        out = self.do1(x).permute(0,2,1)
        out = self.fc2(out).permute(0,2,1)
        out = self.do2(out)
        
        return out+ self.skip_param*initial_x

class KAN2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act, use_poly = False, degree_poly = 3, use_base_update = True, base_activation = F.silu, use_same_fn = False, use_same_weight = False, use_cpd = False, use_softmax_prod = False, num_grids = 8, skip_param = 1, init = 'default',spline_weight_init_scale = 0.1, grid = [-1,1]):
        super(KAN2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.skip_param = skip_param
        self.fc1 = KANLinear(hidden_size, hidden_c,use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, init = init,spline_weight_init_scale = spline_weight_init_scale, grid_min = grid[0], grid_max = grid[1])
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = KANLinear(hidden_c, hidden_size, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids, init = init,spline_weight_init_scale = spline_weight_init_scale, grid_min = grid[0], grid_max = grid[1])
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        
        #out = self.do1(self.act(self.fc1(self.ln(x))))
        #out = self.do2(self.fc2(out))

        out = self.do1(self.fc1(x))
        out = self.do2(self.fc2(out))
        return out+ self.skip_param*x


class MLP_MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MLP_MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x



if __name__ == '__main__':
    net = KANMixer(
        in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=8, 
        num_classes=10, 
        drop_p=0.,
        off_act=False,
        is_cls_token=True
        )
    torchsummary.summary(net, (3,32,32))
