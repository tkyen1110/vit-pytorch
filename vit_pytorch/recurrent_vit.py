from functools import partial

import torch
from torch import nn, einsum

# https://github.com/arogozhnikov/einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_height = 6,
        window_width = 10
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        self.rel_pos_bias = nn.Embedding((2 * window_height - 1) * (2 * window_width - 1), self.heads)
        pos_h = torch.arange(window_height)
        pos_w = torch.arange(window_width)
        grid = torch.stack(torch.meshgrid(pos_h, pos_w))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos[:,:,0] += window_height - 1
        rel_pos[:,:,1] += window_width - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_width - 1, 1])).sum(dim = -1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
        
        '''
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
        '''


    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        # x.shape = (2,  16, 16, 6, 10, 64)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        # x.shape = (512, 60, 64)
    
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # q.shape = (512, 60, 64)
        
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))
        # q.shape = (512, 2, 60, 32)

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # sim.shape = (512, 2, 60, 60)
        
        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        # sim.shape = (512, 2, 60, 60)

        # attention
        attn = self.attend(sim)
        # attn.shape = (512, 2, 60, 60)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out.shape = (512, 2, 60, 32)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)
        # out.shape = (512, 6, 10, 64)

        # combine heads out
        out = self.to_out(out)
        # out.shape = (512, 6, 10, 64)

        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    def __init__(
        self,
        dim_input = 20,
        dim_conv = 64,
        depth = (1, 1, 1, 1), # support only (1, 1, 1, 1) now
        window_height = 6,
        window_width = 10,
        dim_head = 32,
        dropout = 0.1,
    ):
        super().__init__()

        # variables
        num_stages = len(depth)

        kernels = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        paddings = [2, 1, 1, 1]
        dims = tuple(map(lambda i: (2 ** i) * dim_conv, range(num_stages)))
        dims = (dim_input, *dims)
        # dim_pairs = ((kernel, stride, padding, input_channel, output_channel), ...)
        dim_pairs = tuple(zip(kernels, strides, paddings, dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # iterate through stages
        for (kernel, stride, padding, layer_dim_in, layer_dim_out), layer_depth in zip(dim_pairs, depth):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim_out

                block = nn.Sequential(
                    # (N, C, H, W)       = (2,  20, 384, 640), (2,  64,  96, 160), (2, 128,  48,  80), (2, 256,  24,  40)
                    nn.Conv2d(stage_dim_in, layer_dim_out, kernel_size = kernel, stride = stride, padding = padding),
                    # (N, Cx2, H/2, W/2) = (2,  64,  96, 160), (2, 128,  48,  80), (2, 256,  24,  40), (2, 512,  12,  20)

                    Rearrange('b c (h h1) (w w1) -> b h w h1 w1 c', h1 = window_height, w1 = window_width),  # block-like attention
                    #                      (2,  16, 16, 6, 10, 64)
                    PreNormResidual(layer_dim_out, Attention(dim = layer_dim_out, dim_head = dim_head, dropout = dropout, 
                                                             window_height = window_height, window_width = window_width)),
                    #                      (2,  16, 16, 6, 10, 64)
                    PreNormResidual(layer_dim_out, FeedForward(dim = layer_dim_out, dropout = dropout)),
                    #                      (2,  16, 16, 6, 10, 64)
                    Rearrange('b h w h1 w1 c -> b c (h h1) (w w1)'),
                    #                      (2,  64, 96, 160)
                    
                    Rearrange('b c (h1 h) (w1 w) -> b h w h1 w1 c', h1 = window_height, w1 = window_width),  # grid-like attention
                    #                      (2, 16, 16, 6, 10, 64)
                    PreNormResidual(layer_dim_out, Attention(dim = layer_dim_out, dim_head = dim_head, dropout = dropout, 
                                                             window_height = window_height, window_width = window_width)),
                    #                      (2, 16, 16, 6, 10, 64)
                    PreNormResidual(layer_dim_out, FeedForward(dim = layer_dim_out, dropout = dropout)),
                    #                      (2, 16, 16, 6, 10, 64)
                    Rearrange('b h w h1 w1 c -> b c (h1 h) (w1 w)'),
                    #                      (2,  64, 96, 160)
                )

                self.layers.append(block)

    def forward(self, x):
        print(x.shape)
        for stage in self.layers:
            x = stage(x)
            print(x.shape)
        return x
