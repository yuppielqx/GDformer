import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, sim_l = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, sim_l


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        # sim_ls = torch.zeros((x.shape[0], x.shape[1]))
        sim_ls = torch.zeros((x.shape[0], x.shape[1])).cuda()
        for attn_layer in self.attn_layers:
            x, attn, sim_loss = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
            sim_ls += sim_loss

        if self.norm is not None:
            x = self.norm(x)

        # return x, attns, sim_ls / len(self.attn_layers)
        return x, attns, sim_ls


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, ratio, d_model=512, n_heads=8, e_layers=3
                 , num_proto=10, len_map=10, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding raw data to dimension of transformer
        self.embedding = DataEmbedding(win_size, enc_in, d_model, ratio, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, enc_in, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, num_proto, len_map),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, enc_in, bias=True)

    def forward(self, x):
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - means) / stdev

        enc_out = self.embedding(x) # B, L, D
        enc_out, attns, sim_ls = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        enc_out = enc_out
        enc_out = enc_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, enc_out.shape[1], 1))
        enc_out = enc_out + (means[:, 0, :].unsqueeze(1).repeat(1, enc_out.shape[1], 1))

        return enc_out, attns, sim_ls