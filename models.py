import torch
import torch.nn as nn
import torch.nn.functional as nnf


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "gelu":
        return nn.GELU(approximate='tanh')
    else:
        raise NotImplementedError("Activation not implemented")


class ScaleShift(nn.Module):
    def __init__(self, dim):
        super(ScaleShift, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(dim))
        self.shift = nn.Parameter(torch.Tensor(dim))
        self.scale.data.fill_(1)
        self.shift.data.fill_(0)

    def forward(self, x):
        return x * self.scale + self.shift


class InstanceNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super(InstanceNorm, self).__init__()
        self.affine = affine

        if affine:
            self.to_x = ScaleShift(dim)

    def forward(self, x):
        assert len(x.shape) == 3, 'Input must be of shape (batch_size, seq_len, input_dim)'
        x = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)

        if self.affine:
            return self.to_x(x)

        return x


class SwiGLU(nn.Module):
    def __init__(self, dim, affine=True):
        super(SwiGLU, self).__init__()
        self.dim = dim
        self.affine = affine

        if affine:
            self.to_x = ScaleShift(dim)
            self.to_gate = ScaleShift(dim)

    def forward(self, x):
        assert x.shape[-1] / 2 == self.dim, ''
        x, gate = x.chunk(2, dim=-1)

        if self.affine:
            return nnf.silu(self.to_gate(gate)) * self.to_x(x)

        return nnf.silu(gate) * x


class _FeedForward(nn.Module):
    def __init__(self, hidden_dim, expand_dim, activation, dropout):
        super(_FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expand_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expand_dim, hidden_dim)
        )

    def forward(self, x):
        return self.ff(x)


# Not Used
class _MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.to_keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, queries, keys, values, mask=None):
        batch_size, seq_len, embed_dim = values.shape
        head_dim = embed_dim // self.n_heads

        queries = self.to_queries(queries).view(batch_size, seq_len, self.n_heads, head_dim)
        keys = self.to_keys(keys).view(batch_size, seq_len, self.n_heads, head_dim)
        values = self.to_values(values).view(batch_size, seq_len, self.n_heads, head_dim)

        queries = queries.transpose(1, 2).contiguous().view(batch_size * self.n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2).contiguous().view(batch_size * self.n_heads, seq_len, head_dim)
        values = values.transpose(1, 2).contiguous().view(batch_size * self.n_heads, seq_len, head_dim)

        queries = queries / (embed_dim ** (1 / 4))
        keys = keys / (embed_dim ** (1 / 4))

        out = torch.bmm(queries, keys.transpose(1, 2))
        out = torch.softmax(out, dim=-1)
        out = torch.bmm(out, values).view(batch_size, self.n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * head_dim)
        return self.to_out(out)


class _EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, expand_dim, n_heads, activation, dropout):
        super(_EncoderLayer, self).__init__()
        self.norm1 = InstanceNorm(hidden_dim, affine=True)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, bias=False, dropout=dropout)
        self.norm2 = InstanceNorm(hidden_dim, affine=True)
        self.ff = _FeedForward(hidden_dim, expand_dim, activation, dropout)

    def forward(self, x):
        self.norm1(x)
        x = self.attn(x, x, x, is_causal=True)[0] + x
        self.norm2(x)
        return self.ff(x) + x


class Embedder(nn.Module):
    def __init__(self, a2v_dim, lm, activation='gelu', dropout=0.1):
        super(Embedder, self).__init__()
        self.lm = lm
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(384, a2v_dim),
            get_activation(activation),
            nn.Linear(a2v_dim, a2v_dim)
        )

    def forward(self, x, mask):
        desc = self.lm(input_ids=x, attention_mask=mask)[0]
        return self.head(desc[:, 0, :])


class TSEncoderBlock(nn.Module):
    def __init__(self, input_dims, hidden_dim, n_encoders, n_heads, expand_dim=4, activation='gelu', dropout=0.1):
        super(TSEncoderBlock, self).__init__()
        self.encoders = nn.Sequential(
            *[_EncoderLayer(hidden_dim, expand_dim, n_heads, activation, dropout)
              for _ in range(n_encoders)]
        )

        self.linear = nn.Linear(hidden_dim, input_dims[-1])

    def forward(self, x):
        x = self.encoders(x)
        return self.linear(x)
