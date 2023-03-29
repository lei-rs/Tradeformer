from transformers import RobertaModel, RobertaConfig
import torch.nn.functional as nnf
import torch.nn as nn
import torch
import math


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


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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


class _MLP(nn.Module):
    def __init__(self, hidden_dim, n_mlp, activation, dropout):
        super(_MLP, self).__init__()
        modlist = []
        for _ in range(n_mlp):
            modlist += [nn.Linear(hidden_dim, hidden_dim),
                        get_activation(activation),
                        nn.Dropout(dropout)]

        self.mlp = nn.Sequential(*(modlist + [nn.Linear(hidden_dim, hidden_dim)]))

    def forward(self, x):
        return self.mlp(x)


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
        self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.attn = _MultiheadAttention(hidden_dim, n_heads)
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        self.ff = _FeedForward(hidden_dim, expand_dim, activation, dropout)

    def forward(self, x):
        self.norm1(x)
        x = self.attn(x, x, x) + x
        self.norm2(x)
        return self.ff(x) + x


class NLPEmbedder(nn.Module):
    def __init__(self, metadata, nlp_dims, total, a2v_dim, activation='gelu', dropout=0.1):
        super(NLPEmbedder, self).__init__()
        hidden_dim, n_heads, num_layers = nlp_dims
        self.register_buffer('descriptions', metadata.descriptions)
        self.register_buffer('masks', metadata.masks)
        roberta_config = RobertaConfig(
            vocab_size=16000,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=n_heads,
            intermediate_size=hidden_dim * 4,
            type_vocab_size=1
        )
        self.lm = RobertaModel(config=roberta_config, add_pooling_layer=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, a2v_dim),
            nn.Dropout(dropout)
        )
        self.register_buffer('embeds', torch.zeros((total, a2v_dim)))

    def forward(self, x, ticker):
        desc = self.lm(input_ids=self.descriptions[ticker], attention_mask=self.masks[ticker])[1]
        desc = self.head(desc).unsqueeze(1).repeat(1, x.shape[1], 1)
        return torch.cat((x, desc), dim=-1)

    def compile_embeds(self):
        self.register_buffer('embeds', self.head(self.lm(input_ids=self.descriptions, attention_mask=self.masks)[1]).unsqueeze(1))


class EmbedDict(nn.Module):
    def __init__(self, a2v_dim, total):
        super(EmbedDict, self).__init__()
        self.embed = nn.Parameter(torch.normal(0, 1, (total, a2v_dim)))

    def forward(self, x, ticker):
        embd = self.embed[ticker].unsqueeze(1).repeat(1, x.shape[1], 1)
        return torch.cat((x, embd), dim=-1)

    def compile_embeds(self):
        pass


class _TFEncoderBlock(nn.Module):
    def __init__(self, dims, n_encoders, n_heads, n_mlp, expand_dim=4, activation='gelu',
                 dropout=0.1):
        super(_TFEncoderBlock, self).__init__()
        back_length, n_features, hidden_dim, forward_length = dims
        self.pe = PositionalEmbedding(hidden_dim)
        '''self.encoders = nn.Sequential(
            *[_EncoderLayer(hidden_dim, expand_dim, n_heads, activation, dropout)
              for _ in range(n_encoders)]
        )'''
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * expand_dim, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=n_encoders)
        self.linear = nn.Linear(hidden_dim, 1)
        self.mlp = _MLP(back_length, n_mlp, activation, dropout)
        self.linear2 = nn.Linear(back_length, forward_length)

    def forward(self, x):
        x = self.pe(x)
        x = self.encoders(x)
        x = self.linear(x).squeeze(-1)
        x = self.mlp(x)
        return nnf.tanh(self.linear2(x).squeeze(-1))


class Tradeformer(nn.Module):
    def __init__(self, dims, a2v_dim, n_encoders, n_heads, n_mlp, embd_meth, metadata, total, nlp_dims,
                 expand_dim=4, activation='gelu', dropout=0.1):
        super(Tradeformer, self).__init__()
        if embd_meth == 'nlp':
            self.linear = nn.Linear(dims[1] + a2v_dim, dims[2])
            self.embedder = NLPEmbedder(metadata, nlp_dims, total, a2v_dim, activation, dropout)

        elif embd_meth == 'dict':
            self.linear = nn.Linear(dims[1] + a2v_dim, dims[2])
            self.embedder = EmbedDict(a2v_dim, total)

        else:
            self.linear = nn.Linear(dims[1], dims[2])
            self.embedder = Identity()

        self.TFEncoder = _TFEncoderBlock(dims, n_encoders, n_heads, n_mlp, expand_dim, activation, dropout)

    def forward(self, x, ticker):
        x = self.embedder(x, ticker)
        return self.TFEncoder(x)
