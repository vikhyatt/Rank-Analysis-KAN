# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torch.distributions as dist
from torch.autograd import Variable
from torch.nn.utils.parametrizations import weight_norm

from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale = [0.1],init = 'default', degree = 8,**kw) -> None:
        self.init_scale = init_scale
        self.init = init
        self.degree = degree
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        if self.init == 'default':
            nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale[0])
        elif self.init == 'uniform':
            nn.init.uniform_(self.weight, a= -2*self.init_scale[0], b= 2*self.init_scale[0])
        elif self.init == 'xavier':
            nn.init.xavier_uniform_(self.weight, gain=10*self.init_scale[0])
        elif self.init == 'beta':
            beta_dist = dist.beta.Beta(concentration1 = self.init_scale[0], concentration0 = self.init_scale[1])
            with torch.no_grad():
                self.weight.copy_(beta_dist.sample(self.weight.shape) - (self.init_scale[0]/(self.init_scale[0]+self.init_scale[1])))
        elif self.init == 'gamma':
            gamma_dist = dist.gamma.Gamma(concentration = self.init_scale[0], rate = self.init_scale[1])
            with torch.no_grad():
                self.weight.copy_(gamma_dist.sample(self.weight.shape) - (self.init_scale[0]/self.init_scale[1]))
        elif self.init == 'exponential':
            expo_dist = dist.exponential.Exponential(rate = self.init_scale[0])
            with torch.no_grad():
                self.weight.copy_(expo_dist.sample(self.weight.shape) - 1/self.init_scale[0])
        elif self.init == 'zero':
            torch.nn.init.zeros_(self.weight)
            
        elif self.init == 'orthogonal':
            torch.nn.init.orthogonal_(self.weight)
            
        else:
            raise ValueError('Unsupported Initialization entered')

    def finite_difference(self, degree = 1):
        w_s = self.weight.view(self.out_features, self.in_features // self.degree, self.degree)
        for i in range(degree):
            w_s = (w_s.narrow(2, 1, w_s.size(2)-1) - w_s.narrow(2, 0, w_s.size(2)-1))
        return (w_s ** 2).sum()

class tied_SplineLinear(nn.Module):
    def __init__(self, in_features, out_features: int, init_scale = [0.1], degree = 3 , use_same_weight = True ,bias=False, w_norm = 0, **kw):
        super(tied_SplineLinear, self).__init__()
        self.init_scale = init_scale[0]
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.use_same_weight = use_same_weight
        self.w_norm = w_norm
        #print(f"{degree+1} weights per row in matrix")
        # degree+1 weights per row in matrix
        
        self.weight = Parameter(torch.Tensor(out_features, self.degree + 1))
        self.weighted_sum  = nn.Conv1d(in_features, out_features, 1, bias = False)
        
        #self.fc1 = SplineLinear(self.degree + 1, self.out_features, init_scale = init_scale)
        #self.weighted_sum = Parameter(torch.Tensor(out_features, in_features))
        """
        if w_norm:  
            self.weighted_sum  = weight_norm(nn.Conv1d(in_features, out_features, 1, bias = False)) 
            self.norm_factor = nn.Parameter(torch.tensor(1.0))
        else:
            self.weighted_sum  = nn.Conv1d(in_features, out_features, 1, bias = False)
        """
        #degree weights shared across the entire matrix + (degree+1)'th weight is unique for each row
        #self.weight_deg = Parameter(torch.Tensor(self.degree))
        #self.weight_one = Parameter(torch.Tensor(self.out_features, 1))

        #degree+1 weights shared across the entire matrix
        #self.weight_deg = Parameter(torch.Tensor(self.degree+1))
        
        #if not self.use_same_weight:
        #    self.weighted_sum = Parameter(torch.Tensor(1, out_features, in_features, 1))
            #self.weighted_sum = self.weighted_sum.unsqueeze(-1)
            
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # degree+1 weights per row in matrix
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

        #degree weights shared across the entire matrix + (degree+1)'th weight is unique for each row
        #nn.init.trunc_normal_(self.weight_deg, mean=0, std=self.init_scale)
        #nn.init.trunc_normal_(self.weight_one, mean=0, std=self.init_scale)

        #degree+1 weights shared across the entire matrix
        #nn.init.trunc_normal_(self.weight_deg, mean=0, std=self.init_scale)
        
        #nn.init.trunc_normal_(self.weighted_sum, mean=0, std=self.init_scale)
        nn.init.trunc_normal_(self.weighted_sum.weight, mean=0, std=self.init_scale)
        
    def normalize(self):
        self.weight.data = F.normalize(self.weight.data, p=2, dim=-1)
        
    def forward(self, x):
        # degree+1 weights per row in matrix
        #out = F.linear(x, self.weight, self.bias)
        
        #x = x.permute(*range(x.ndim - 2), -1, -2)
        #x = F.linear(x,self.weighted_sum, self.bias)
        #x = x.permute(*range(x.ndim - 2), -1, -2)
        og_shape = x.shape
        if len(og_shape) == 4:
            x = x.view(og_shape[0]*og_shape[1], og_shape[2], og_shape[3])
            x = self.weighted_sum(x).view(og_shape[0], og_shape[1], -1, og_shape[3])
        else:
            x = self.weighted_sum(x)
        
        #x = torch.einsum('...ji,ij->...i', x, self.weight).contiguous()
        if self.w_norm:
            weight_normed_param = self.norm_factor * self.weight / torch.norm(self.weight)
            x = (x * weight_normed_param).sum(dim=-1)
        else:
            x = (x * self.weight).sum(dim=-1)

        #degree weights shared across the entire matrix + (degree+1)'th weight is unique for each row
        #final_weight = torch.cat((self.weight_deg.repeat(self.out_features, 1), self.weight_one), dim = -1)
        #out = F.linear(x, final_weight, self.bias)

        #degree+1 weights shared across the entire matrix
        #final_weight = self.weight_deg.repeat(self.out_features, 1)
        #out = F.linear(x, final_weight, self.bias)
        
        #out = out.permute(*range(out.ndim - 2), -1, -2).unsqueeze(-2)
        #print(out.shape)
        #out = torch.matmul(out, self.weighted_sum).squeeze(-1).squeeze(-1)
        #print(out.shape)
        #out = torch.einsum('...ji,ij->...i', out, self.weighted_sum).contiguous()
        #out = F.linear(out, self.weighted_sum, self.bias).contiguous()
        return x


class HankelLinear(nn.Module):
    def __init__(self, in_features, out_features: int, init_scale = [0.1], degree = 3 , use_same_weight = True ,bias=False, w_norm = 0, **kw):
        super(HankelLinear, self).__init__()
        self.init_scale = init_scale[0]
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.use_same_weight = use_same_weight
        self.w_norm = w_norm
        
        self.weight = Parameter(torch.Tensor(out_features + in_features - 1, self.degree + 1))
        self.weighted_sum = Parameter(torch.Tensor(out_features, in_features))
        
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        nn.init.trunc_normal_(self.weighted_sum, mean=0, std=self.init_scale)
       
    def normalize(self):
        self.weight.data = F.normalize(self.weight.data, p=2, dim=-1)
        
    def forward(self, x):
        
        hank =  self.weight.unfold(0, self.in_features, 1).permute(0,2,1) 
        hank = self.weighted_sum.unsqueeze(-1) * hank
        if len(x.shape) == 4:
            x = torch.einsum('bhid,oid->bho', x, hank)
        else:
            x = torch.einsum('hid,oid->ho', x, hank)
            
        return x
        
class tied_SplineLinear_FAST(nn.Module):
    def __init__(self, in_features, out_features: int, init_scale = [0.1], degree = 3 , use_same_weight = True ,bias=False, w_norm = 0, **kw):
        super(tied_SplineLinear_FAST, self).__init__()
        self.init_scale = init_scale[0]
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.w_norm = w_norm
        self.use_same_weight = use_same_weight
        
        self.weight = Parameter(torch.Tensor(out_features, self.degree + 1))
        self.weighted_sum = Parameter(torch.Tensor(out_features, in_features))
        
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        nn.init.trunc_normal_(self.weighted_sum, mean=0, std=self.init_scale)
        
    def normalize(self):
        self.weight.data = F.normalize(self.weight.data, p=2, dim=-1)

    def finite_difference(self, degree = 1):
        w_s = self.weight.view(self.out_features, self.degree+1)
        for i in range(degree):
            w_s = (w_s.narrow(1, 1, w_s.size(1)-1) - w_s.narrow(1, 0, w_s.size(1)-1))
        return (w_s ** 2).sum()
        
    def forward(self, x):
        weights =  self.weight.unsqueeze(-2).expand(self.out_features, self.in_features, self.degree + 1) 
        weights = self.weighted_sum.unsqueeze(-1) * weights
        if len(x.shape) == 4:
            x = torch.einsum('bhid,oid->bho', x, weights)
        else:
            x = torch.einsum('hid,oid->ho', x, weights)
        return x

        #degree weights shared across the entire matrix + (degree+1)'th weight is unique for each row
        #final_weight = torch.cat((self.weight_deg.repeat(self.out_features, 1), self.weight_one), dim = -1)
        #out = F.linear(x, final_weight, self.bias)

        #degree+1 weights shared across the entire matrix
        #final_weight = self.weight_deg.repeat(self.out_features, 1)
        #out = F.linear(x, final_weight, self.bias)
        
        #out = out.permute(*range(out.ndim - 2), -1, -2).unsqueeze(-2)
        #print(out.shape)
        #out = torch.matmul(out, self.weighted_sum).squeeze(-1).squeeze(-1)
        #print(out.shape)
        #out = torch.einsum('...ji,ij->...i', out, self.weighted_sum).contiguous()
        #out = F.linear(out, self.weighted_sum, self.bias).contiguous()
        return x

def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

class circ_layer(nn.Module):
    def __init__(self, in_features: int, init_scale: float = 0.1, **kw) -> None:
        super(circ_layer, self).__init__()
        self.init_scale = init_scale
        self.in_features = in_features
        self.weight = nn.Parameter(torch.rand(in_features))

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

    def forward(self, x):
        self.weight_clone = circulant(self.weight, 0)
        #mm_cx = torch.matmul(x, self.weight_clone)
        mm_cx = F.linear(x, self.weight_clone)
        out = (mm_cx * x) + mm_cx + x
        return out
    

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        grid_type: str = 'uniform',
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        if grid_type == 'uniform':
            grid = torch.linspace(grid_min, grid_max, num_grids)
        elif grid_type == 'chebyshev':
            nodes = torch.cos((2 * torch.arange(1, num_grids + 1) - 1) * math.pi / (2 * num_grids))
            grid = 0.5 * (grid_max - grid_min) * nodes + 0.5 * (grid_max + grid_min)
            
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        if denominator == 0:
            self.denominator = (grid_max - grid_min) / (num_grids - 1)
        elif denominator < 0:
            self.denominator = torch.abs(denominator*self.grid).clone()
            self.denominator = torch.clamp(self.denominator, min = 0.75)
            self.denominator = torch.nn.Parameter(self.denominator, requires_grad=False)
        else:
            self.denominator = denominator

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class PolyBasisFunction(nn.Module):
    def __init__(
        self,
        n: int = 3,
    ):
        super().__init__()
        degree = torch.tensor([i for i in range(n+1)])
        self.degree = torch.nn.Parameter(degree, requires_grad=False)

    def forward(self, x):
        return torch.clamp(torch.pow(x[...,None], self.degree), min = -2, max = 2)

class FourierLayer(nn.Module):
    def __init__(self, input_dim, output_dim,N, P):
        super(FourierLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N
        self.P = P
        self.PI = torch.acos(torch.zeros(1)).item() * 2
        self.fourier_sin_coeffs = nn.Parameter(torch.randn(output_dim, self.N + 1))
        self.fourier_cos_coeffs = nn.Parameter(torch.randn(output_dim, self.N + 1))
        self.trig_coeffs = (2 * self.PI * torch.arange(self.N + 1)) / self.P
        
    def forward(self, x):
        trig_inp = x.unsqueeze(-1) * self.trig_coeffs.view(*[1] * x.dim(), self.N + 1).to(x.device)
        if len(trig_inp.shape) == 4:
            trig_sin = torch.einsum("bhin,on->bhi" , torch.sin(trig_inp), self.fourier_sin_coeffs)
            trig_cos = torch.einsum("bhin,on->bhi" , torch.cos(trig_inp), self.fourier_cos_coeffs)
        else:
            trig_sin = torch.einsum("hin,on->hi" , torch.sin(trig_inp), self.fourier_sin_coeffs)
            trig_cos = torch.einsum("hin,on->hi" , torch.cos(trig_inp), self.fourier_cos_coeffs)
        
        return trig_sin + trig_cos

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        use_poly = False,
        degree_poly = 3,
        use_same_fn = False,
        use_same_weight = False,
        use_cpd = False,
        use_hankel = False,
        use_softmax_prod = False,
        init = 'default',
        grid_type = 'uniform',
        denominator = None,
        w_norm = 0,
        use_fourier = False,
        N = 5,
        P = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_same_fn = use_same_fn
        self.use_same_weight = use_same_weight
        self.use_poly = use_poly
        self.use_softmax_prod = use_softmax_prod
        self.layernorm = None
        self.use_fourier = use_fourier
        
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)

        if use_poly and not use_hankel:
            self.rbf = PolyBasisFunction(degree_poly)
            if not use_same_fn:
                self.spline_linear = SplineLinear(input_dim * (degree_poly+1), output_dim, spline_weight_init_scale, init = init, degree = (degree_poly+1))
            else:
                if self.use_same_weight:
                    self.spline_linear = SplineLinear(degree_poly + 1, output_dim, init_scale = spline_weight_init_scale, init = init)
                else:
                    self.spline_linear = tied_SplineLinear_FAST(input_dim, output_dim, init_scale = spline_weight_init_scale, 
                                                       degree = degree_poly , use_same_weight = use_same_weight, w_norm = w_norm)
                

        elif not use_hankel:
            self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids, grid_type = grid_type, denominator = denominator)
            if not use_same_fn:
                self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale, init = init, degree = num_grids)
            else:
                if self.use_same_weight:
                    self.spline_linear = SplineLinear(num_grids, output_dim, init_scale = spline_weight_init_scale, init = init)
                else:
                    self.spline_linear = tied_SplineLinear_FAST(input_dim, output_dim, init_scale = spline_weight_init_scale, 
                                                       degree = num_grids - 1, use_same_weight = use_same_weight, w_norm = w_norm)
                    
        else:
            self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids, grid_type = grid_type, denominator = denominator)
            self.spline_linear = HankelLinear(input_dim, output_dim, init_scale = spline_weight_init_scale, 
                                                       degree = num_grids - 1, use_same_weight = use_same_weight, w_norm = w_norm)
            
        self.cpd = use_cpd
        if use_same_fn and self.cpd:
            self.circ = circ_layer(output_dim, spline_weight_init_scale)

        if use_fourier:
            self.fourier_layer = FourierLayer(input_dim, output_dim, N = N, P = P)
            
            
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            x = self.layernorm(x)
            
        spline_basis = self.rbf(x)
        if self.use_same_fn:
            ret = self.spline_linear(spline_basis)
            if self.use_same_weight:
                ret = ret.mean(-2)
                
            #assert (self.cpd != self.use_softmax_prod), "Cannot use both CPD and Softmax Product"
            if self.cpd:
                ret = self.circ(ret)

            if self.use_softmax_prod:
                ret = ret * F.softmax(ret, dim = -1)
        else:
            ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))

        if self.use_fourier:
            ret = ret + self.fourier_layer(x)
            
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret
        
    def normalize(self):
        self.spline_linear.normalize()

    def finite_difference(self, degree = 1):
        return self.spline_linear.finite_difference(degree = degree)
        
    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
    ) -> torch.Tensor:         

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
        att = (wq * wk).sum(-1).softmax(-2)     # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
        o = (att[..., None] * wv).sum(-3)        # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)           # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o
