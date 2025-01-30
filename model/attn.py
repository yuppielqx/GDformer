import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import pandas as pd

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def my_js_loss(p, q):
    m = 0.5 * (p + q)
    return 0.5 * my_kl_loss(p, m) + 0.5 * my_kl_loss(q, m)

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, input_c, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)


    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        #self-attention
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores # B, H, L, L

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        return V.contiguous(), series


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, num_proto, len_map, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.num_proto = num_proto
        self.len_map = len_map
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.keys = torch.nn.Parameter(torch.randn(num_proto, d_model))#
        self.values = torch.nn.Parameter(torch.randn(num_proto, d_model))
        self.attn_map = torch.nn.Parameter(torch.randn(len_map, num_proto))#

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        N, _ = self.keys.shape
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)

        keys = self.keys.view(1, N, H, -1).repeat(B, 1, 1, 1)
        values = self.values.view(1, N, H, -1).repeat(B, 1, 1, 1)

        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )

        attn_map = torch.softmax(self.attn_map, dim=-1).view(1, 1, self.len_map, -1).repeat(B, H, 1, 1)
        sim = torch.einsum("bhln,bhmn->bhlm", attn, attn_map)
        sim_l = torch.sum(torch.sum(sim, dim=-1), dim=1)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn, sim_l




