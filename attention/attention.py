import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PretrainedConfig
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.attention_head_size = self.dim // self.n_heads

        if self.dim % self.n_heads != 0:
            raise ValueError(f"n_heads={self.n_heads} must divide dim={self.dim}")

        self.sketch = getattr(config, "sketch", False)
        self.sketch_dim = getattr(config, "sketch_dim", self.dim)
        self.sketch_matrix = getattr(config, "sketch_matrix", None)
        self.sketch_attention = getattr(config, "sketch_attention", None)

        if self.sketch:
            self.A = nn.Parameter(torch.randn(self.dim, self.sketch_dim))  # (d, k)
            self.B = nn.Parameter(torch.randn(self.dim, self.sketch_dim))
            self.C = nn.Parameter(torch.randn(self.dim, self.sketch_dim))
        else:
            self.q_lin = nn.Linear(self.dim, self.dim)
            self.k_lin = nn.Linear(self.dim, self.dim)
            self.v_lin = nn.Linear(self.dim, self.dim)

        if self.sketch_attention == "learned":
            self.proj_E = nn.Linear(config.max_position_embeddings, self.sketch_dim)
            self.proj_F = nn.Linear(config.max_position_embeddings, self.sketch_dim)

        self.out_lin = nn.Linear(self.dim, self.dim)

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        bs, q_len, _ = query.size()
        k_len = key.size(1)
        dim_per_head = self.attention_head_size

        def shape(x): return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        def unshape(x): return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        if self.sketch:
            q = torch.matmul(query, self.A)
            k = torch.matmul(key, self.B)
            v = torch.matmul(value, self.C)
        else:
            q = self.q_lin(query)
            k = self.k_lin(key)
            v = self.v_lin(value)

        q = shape(q) / math.sqrt(dim_per_head)
        k = shape(k)
        v = shape(v)

        if self.sketch_attention == "learned":
            # Project over seq_len
            k_proj = self.proj_E(k.transpose(-2, -1)).transpose(-2, -1)  # (bs, heads, k_len, k_sketch)
            v_proj = self.proj_F(v.transpose(-2, -1)).transpose(-2, -1)  # (bs, heads, k_sketch, dim)
        elif self.sketch_attention == "random":
            S = self.sketch_matrix  # (sketch_dim, k_len)
            if S is None or S.shape != (self.sketch_dim, k_len):
                raise ValueError(f"sketch_matrix must be of shape ({self.sketch_dim}, {k_len})")
            k_proj = torch.matmul(k, S.T)  # (bs, heads, d, sketch)
            v_proj = torch.matmul(S, v)    # (bs, heads, sketch, d)
        else:
            k_proj = k
            v_proj = v


        if self.sketch_attention in {"learned", "random"}:
            scores = torch.matmul(q, k_proj.transpose(-2, -1))  # (bs, heads, q_len, sketch_dim)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))  # (bs, heads, q_len, k_len)

        mask = (mask == 0).unsqueeze(1).unsqueeze(2).expand_as(scores)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask


        if self.sketch_attention in {"learned", "random"}:
            context = torch.matmul(weights, v_proj)
        else:
            context = torch.matmul(weights, v)

        context = unshape(context)
        output = self.out_lin(context)

        return (output, weights) if output_attentions else (output,)
