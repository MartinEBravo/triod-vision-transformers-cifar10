# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch.utils.data import DataLoader
from torch import nn
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from triod.utils import generate_structured_masked_x, SequentialWithP, test_prefix_od
from triod.layers.linear import TriODLinear
from triod.layers.layer_norm import TriODHeadLayerNorm

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, n_head, fn, triangular: bool = False):
        super().__init__()
        self.norm = TriODHeadLayerNorm(dim, n_head=n_head, triangular=triangular)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., triangular: bool = False):
        super().__init__()
        self.fc1 = TriODLinear(dim, hidden_dim, triangular=triangular)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = TriODLinear(hidden_dim, dim, triangular=triangular)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, p: float | None = None):
        x = self.fc1(x, p=p)     
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, p=None) 
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., triangular: bool = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = TriODLinear(dim, inner_dim * 3, blocks=3, triangular=triangular)

        self.to_out = SequentialWithP(
            TriODHeadLayerNorm(inner_dim, n_head=heads, triangular=True),
            TriODLinear(inner_dim, dim, triangular=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, p: float | None = None):
        h_keep = self.heads if p is None else max(1, math.ceil(self.heads * p))
        keep = h_keep * self.dim_head
        p_inner = keep / self.inner_dim



        qkv = self.to_qkv(x, p=p_inner).chunk(3, dim=-1) 
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=h_keep, d=self.dim_head) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out, p=None)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., triangular: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, heads, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, triangular=triangular), triangular=triangular),
                PreNorm(dim, heads, FeedForward(dim, mlp_dim, dropout = dropout, triangular=triangular), triangular=triangular)
            ]))
    def forward(self, x, p: float | None = None):
        for attn, ff in self.layers:
            x = attn(x, p=p) + x
            x = ff(x, p=p) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., triangular: bool = False, p_s: list[float] | None = None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        self.dim = dim
        self.heads = heads

        self.p_s = p_s

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            TriODLinear(patch_dim, dim, triangular=False),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, triangular=triangular)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = SequentialWithP(
            TriODHeadLayerNorm(dim, n_head=heads, triangular=triangular),
            TriODLinear(dim, num_classes, triangular=False)
        )

    def forward(self, img, p=None, return_prelast=False, all_models=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # # # TODO: temporal fix, we should use TriODEmbedding with p in the future but they are equivalent for now
        # if p is not None:
        #     head_dim = self.dim // self.heads
        #     keep_heads = max(1, math.ceil(self.heads * p))
        #     keep_dim = keep_heads * head_dim
        #     p = keep_dim / self.dim
        #     x = x[:, :, :keep_dim]

        x = self.transformer(x, p=p)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)


        # For testing intermediate representations
        if return_prelast:
            return x

        if all_models and self.p_s is not None:
            x = generate_structured_masked_x(x, self.p_s)

        x = x.unsqueeze(1)

        return self.mlp_head(x,p=None).squeeze(1)

if __name__ == '__main__':
    model = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 128,
        depth = 6,
        heads = 8,
        mlp_dim = 256,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.1,
        triangular = True,
        p_s = [0.25, 0.5, 0.75, 1.0]
    )

    img = torch.randn(10, 3, 32, 32)
    dataloader = DataLoader([(img, img)])
    logits = model(img, p=0.5)
    print(logits.shape)
    test_prefix_od(model, 'cuda', dataloader, [0.25, 0.5, 0.75, 1.0])