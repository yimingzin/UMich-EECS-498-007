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

    attention_scores = torch.zeros((N, K, K_k), dtype=query.dtype, device=key.device)
    attention_scores = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(M)

    if mask is not None:
        attention_scores = torch.masked_fill(attention_scores, mask, -1e9)

    weights_softmax = F.softmax(attention_scores, dim=-1)
    y = torch.bmm(weights_softmax, value)

    return y, weights_softmax

class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()
        self.query = nn.Linear(dim_in, dim_q)
        self.key = nn.Linear(dim_in, dim_q)
        self.value = nn.Linear(dim_in, dim_v)
        self.weights_softmax = None

        for layer in [self.query, self.key, self.value]:
            Dim_in, Dim_out = layer.weight.shape
            c = math.sqrt(6 / (Dim_in + Dim_out))
            nn.init.uniform_(layer.weight, a=-c, b=c)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        y = None
        self.weights_softmax = (
            None
        )
        query = self.query.forward(query)
        key = self.key.forward(key)
        value = self.value.forward(value)

        y, self.weights_softmax = scaled_dot_product_no_loop_batch(query, key, value, mask)

        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        self.heads = nn.ModuleList([SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)])
        self.linear_multihead = nn.Linear(dim_out * num_heads, dim_in)

        c = math.sqrt(6 / (dim_out * num_heads + dim_in))
        nn.init.uniform_(self.linear_multihead.weight, a=-c, b=c)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        y = []
        for head in self.heads:
            y.append(head.forward(query, key, value, mask))
        y = torch.cat(y, dim=-1)

        y = self.linear_multihead.forward(y)

        return y

class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor):

        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / std

        y = self.gamma * x_norm + self.beta

        return y

class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        self.linear_1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim_feedforward, inp_dim)

        c = math.sqrt(6 / (inp_dim + hidden_dim_feedforward))
        for layer in [self.linear_1, self.linear_2]:
            nn.init.uniform_(layer.weight, a=-c, b=c)

    def forward(self, x):

        y = self.linear_1.forward(x)
        y = self.relu.forward(y)
        y = self.linear_2.forward(y)

        return y

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(f"""The value emb_dim = {emb_dim} is not divisible by num_heads = {num_heads}""")

        self.headsAttention = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.layer_normalization_1 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.layer_normalization_2 = LayerNormalization(emb_dim)

    def forward(self, x: Tensor):

        y = self.dropout.forward(self.layer_normalization_1.forward(self.headsAttention.forward(x, x, x) + x))
        y = self.dropout.forward(self.layer_normalization_2.forward(self.feedforward.forward(y) + y))

        return y
