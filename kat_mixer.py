import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchsummary
from fastkan import FastKAN as KAN
from fastkan import FastKANLayer as KANLinear
from kat import KAT

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



class KATMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False, use_poly = False, degree_poly = 3, use_base_update = True, base_activation = F.silu, use_same_fn = False, use_same_weight = False, use_pe = False, use_cpd = False, use_softmax_prod = False, num_grids = 8, skip_min = 1):
        super(KATMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        print(f"Polynomial Basis: {use_poly}, Degree of Polynomial: {degree_poly}, Using Same Function: {use_same_fn}, Using same weights: {use_same_weight}, Positional Embeddings: {use_pe}, CPD Decomposition: {use_cpd}, Softmax Prod: {use_softmax_prod}")

        self.patch_emb = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            KANLinear((patch_size ** 2) * in_channels, hidden_size, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids)
        )

        self.use_pe = use_pe
        if use_pe:
            self.pe = get_sinusoid_encoding(num_patches, hidden_size)


        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        delta_skip = (1-skip_min)/num_layers
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, skip_param = (1 - delta_skip*i)) 
            for i in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.clf = KANLinear(hidden_size, num_classes, use_poly = use_poly, degree_poly = degree_poly, use_base_update = use_base_update, base_activation = base_activation, use_same_fn = use_same_fn, use_same_weight = use_same_weight, use_cpd = use_cpd, use_softmax_prod = use_softmax_prod, num_grids = num_grids)


    def forward(self, x):
        out = self.patch_emb(x)

        #Use Positional Embeddings
        if self.use_pe:
            if self.pe.device != out.device:
                self.pe = self.pe.to(out.device)
                
            out = out + self.pe
            
        
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        
        #out = self.mlp_mixer_layers(out)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        return out



class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, skip_param = 1):
        super(MixerLayer, self).__init__()
        self.kan1 = KAN1(num_patches, hidden_s, drop_p, skip_param = skip_param)
        self.kan2 = KAN2(hidden_size, hidden_c, drop_p, skip_param = skip_param)
    def forward(self, x):
        out = self.kan1(x)
        out = self.kan2(out)
        return out

class KAN1(nn.Module):
    def __init__(self, num_patches, hidden_s, drop_p, skip_param = 1):
        super(KAN1, self).__init__()
        self.skip_param = skip_param
        self.kan = KAT(in_features = num_patches, hidden_features = hidden_s, out_features = num_patches, norm_layer = nn.LayerNorm, drop = drop_p)
    
    def forward(self, x):
        out = x.permute(0,2,1)
        out = self.kan(out).permute(0,2,1)
        return out + self.skip_param * x

class KAN2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, skip_param = 1):
        super(KAN2, self).__init__()
        self.skip_param = skip_param
        self.kan = KAT(in_features = hidden_size, hidden_features = hidden_c, out_features = hidden_size, norm_layer = nn.LayerNorm, drop = drop_p)
        
    def forward(self, x):
        return self.kan(x) + self.skip_param*x



if __name__ == '__main__':
    net = KATMixer(
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
