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
        input_str: str, token_dict: dict, spe_tokens: list
):
    out = []

    words = input_str.split()
    for i, w in enumerate(words):
        if w in spe_tokens:
            out.append(token_dict[w])
        else:
            for ch in w:
                out.append(token_dict[ch])

    return out

def scaled_dot_product_two_loop_single(
        query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

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
) -> Tensor:

    N, K, M = query.shape
    N, K_k, M = key.shape

    attention_scores = torch.zeros((N, K, K_k), dtype=query.dtype, device=query.device)

    for i in range(K):
        for j in range(K_k):
            attention_scores[:, i, j] = torch.sum(query[:, i, :] * key[:, j, :], dim=1)

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
            Dim_in, Dim_out = layer.weight.shape
            c = math.sqrt(6 / (Dim_in + Dim_out))
            nn.init.uniform_(layer.weight, a=-c, b=c)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        y = None
        self.weights_softmax = (
            None
        )
        query = self.q.forward(query)
        key = self.k.forward(key)
        value = self.v.forward(value)

        y, self.weights_softmax = scaled_dot_product_no_loop_batch(query, key, value, mask)

        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)])
        self.output_transform = nn.Linear(num_heads * dim_out, dim_in)

        c = math.sqrt(6 / (dim_in + dim_out))
        nn.init.uniform_(self.output_transform.weight, a=-c, b=c)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        y = []
        for head in self.heads:
            y.append(head.forward(query, key, value, mask))
        y = torch.cat(y, dim=-1)
        y = self.output_transform(y)

        return y

class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor):

        mean = torch.mean(x, dim = -1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)

        norm = (x - mean) / std

        y = self.gamma * norm + self.beta

        return y

class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()
        self.linear_1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim_feedforward, inp_dim)

        c = math.sqrt(6 / (inp_dim + hidden_dim_feedforward))
        nn.init.uniform_(self.linear_1.weight, a=-c, b=c)
        nn.init.uniform_(self.linear_2.weight, a=-c, b=c)

    def forward(self, x: Tensor):
        y = self.linear_2.forward(self.relu.forward(self.linear_1.forward(x)))
        return y

class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible by num_heads = {num_heads}. Please select an appropriate value."""
            )

        self.MultiHeadAttention = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.ln_1 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.ln_2 = LayerNormalization(emb_dim)

    def forward(self, x):
        x = self.dropout.forward(self.ln_1.forward(self.MultiHeadAttention.forward(x, x, x) + x))
        y = self.dropout.forward(self.ln_2.forward(x + self.feedforward.forward(x)))

        return y
