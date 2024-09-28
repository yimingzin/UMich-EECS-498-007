import math

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

def generate_token_dict(vocab):

    token_dict = {}
    for i, w in enumerate(vocab):
        token_dict[w] = i

    return token_dict

def prepocess_input_sequence(
    input_str: str, token_dict: dict, spc_token: list
):

    y = []
    words = input_str.split()
    for i, w in enumerate(words):
        if w in spc_token:
            y.append(token_dict[w])
        else:
            for ch in w:
                y.append(token_dict[ch])

    return y

def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
):

    K, M = query.shape
    K_k, M = key.shape

    attention_scores = torch.zeros((K, K_k), dtype=query.dtype, device=query.device)

    for i in range(K):
        for j in range(K_k):
            attention_scores[i, j] = torch.inner(query[i], key[j])

    attention_scores = attention_scores / math.sqrt(M)
    weights_softmax = F.softmax(attention_scores, dim=-1)
    y = torch.mm(weights_softmax, value)

    return y

def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
):

    N, K, M = query.shape
    N, K_k, M = key.shape

    attention_scores = torch.zeros((N, K, K_k), dtype=query.dtype, device=query.device)

    for i in range(K):
        for j in range(K_k):
            attention_scores[:, i, j] = torch.sum(query[:, i, :] * key[:, j, :], dim=-1)

    attention_scores = attention_scores / math.sqrt(M)
    weights_softmax = F.softmax(attention_scores, dim=-1)
    y = torch.bmm(weights_softmax, value)

    return y

def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
):

    N, K, M = query.shape
    N, K_k, M = key.shape

    attention_scores = torch.zeros((N, K, K_k), dtype=query.dtype, device=query.device)

    attention_scores = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(M)
    if mask is not None:
        attention_scores = torch.masked_fill(attention_scores, mask, -1e9)

    weights_softmax = F.softmax(attention_scores, dim=-1)
    y = torch.bmm(weights_softmax, value)

    return y, weights_softmax

class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)
        self.weights_softmax = None

        for layer in [self.q, self.k, self.v]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        y = None
        self.weights_softmax = (
            None
        )
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        y, self.weights_softmax = scaled_dot_product_no_loop_batch(query, key, value, mask)

        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        self.heads = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)
        ])
        self.linear_mult = nn.Linear(num_heads * dim_out, dim_in)

        nn.init.xavier_uniform_(self.linear_mult.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        y = []
        for head in self.heads:
            y.append(head(query, key, value, mask))
        y = torch.cat(y, dim=-1)
        y = self.linear_mult(y)

        return y

class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / (std + self.epsilon)

        y = self.gamma * x_norm + self.beta

        return y

class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        self.linear_1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim_feedforward, inp_dim)

        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor):

        y = self.linear_2(self.relu(self.linear_1(x)))
        return y

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(f"""The value emb_dim = {emb_dim} is not divisible by num_heads = {num_heads}""")

        self.attention = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.layer_norm_1 = LayerNormalization(emb_dim)
        self.layer_norm_2 = LayerNormalization(emb_dim)
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out_1 = self.dropout(self.layer_norm_1(self.attention(x, x, x) + x))
        out_2 = self.dropout(self.layer_norm_2(self.feed_forward(out_1) + out_1))

        return out_2

def get_subsequent_mask(seq):

    N, K = seq.shape

    mask = torch.tril(torch.ones((N, K, K), dtype=seq.dtype, device=seq.device))
    mask = (mask == 0)

    return mask

class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(f"""the value emb_dim = {emb_dim} is not divisible by num_heads = {num_heads}""")

        self.attention_self = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.attention_cross = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.layer_norm_1 = LayerNormalization(emb_dim)
        self.layer_norm_2 = LayerNormalization(emb_dim)
        self.layer_norm_3 = LayerNormalization(emb_dim)
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor = None):

        out_1 = self.dropout(self.layer_norm_1(self.attention_self(dec_inp, dec_inp, dec_inp, mask) + dec_inp))
        out_2 = self.dropout(self.layer_norm_2(self.attention_cross(out_1, enc_inp, enc_inp) + out_1))
        out_3 = self.dropout(self.layer_norm_3(self.feed_forward(out_2) + out_2))

        return out_3

class Encoder(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, src_seq: Tensor):

        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq

class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, num_layers: int, dropout: float, vocab_len: int):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):
        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out

