import math

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


def generate_token_dict(vocab):
    """
    :param vocab:  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
    :return: dictionary token_dict: vocab列表中的元素对应的索引 {"0": 0, "1": 1, ... "POSITIVE": 10, ...}
    """
    token_dict = {}

    for i, w in enumerate(vocab):
        token_dict[w] = i

    return token_dict

def prepocess_input_sequence(
        input_str: str, token_dict: dict, spc_tokens: list
):
    """
    :param input_str: eg: "BOS POSITIVE 0333 add POSITIVE 0696 EOS"
    :param token_dict: eg: {'BOS': 1, 'POSITIVE': 2, 'add': 3, 'EOS': 4, '0': 5, '3': 6, '6': 7, '9': 8}
    :param spc_tokens: eg: ['BOS', 'POSITIVE', 'add', 'EOS']
    :return: [1, 2, 5, 6, 6, 6, 3, 2, 5, 7, 8, 7, 4]
    """
    out = []

    words = input_str.split()

    for i, w in enumerate(words):
        if w in spc_tokens:
            out.append(token_dict[w])
        else:
            for ch in w:
                out.append(token_dict[ch])

    return out

def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
):

    K, M = query.shape
    K_k, M = key.shape

    attention_scores = torch.zeros((K, K_k), dtype=query.dtype, device=query.device)

    for i in range(K):
        for j in range(K_k):
            attention_scores[i, j] = torch.inner(query[i], key[j])

    '''
    # with no loops dot_product_single
    attention_scores = torch.mm(query, key.transpose(1, 0))
    '''

    attention_scores = attention_scores / math.sqrt(M)
    attention_scores = F.softmax(attention_scores, dim=-1)

    out = torch.mm(attention_scores, value)

    return out

def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
):

    N, K, M = query.shape
    N, K_k, M = key.shape

    attention_scores = torch.zeros((N, K, K_k), dtype=query.dtype, device=query.device)

    for i in range(K):
        for j in range(K_k):
            attention_scores[:, i, j] = (query[:, i] * key[:, j]).sum(dim=-1)
    '''
    # with no loops dot_product_batch
    attention_scores = torch.bmm(query, key.permute(0, 2, 1))
    '''

    attention_scores = attention_scores / math.sqrt(M)
    attention_scores = F.softmax(attention_scores, dim=-1)

    out = torch.bmm(attention_scores, value)

    return out

def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
):

    N, K, M = query.shape
    N, K_k, M = key.shape
    y = None
    weights_softmax = None

    attention_scores = torch.bmm(query, key.permute(0, 2, 1))
    attention_scores = attention_scores / math.sqrt(M)

    if mask is not None:
        torch.masked_fill(attention_scores, mask, -1e9)

    weights_softmax = F.softmax(attention_scores, dim=-1)
    y = torch.bmm(weights_softmax, value)

    return y, weights_softmax

class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()
        """
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and ley vector
            dim_v: an int value for output dimension for value vectors
        """
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)

        """
        If a Linear layer has input dimension D_in and output dimension D_out  
        then initialize the weights sampled from a uniform distribution bounded
        by [-c, c]                                                             
        where c = sqrt(6/(D_in + D_out)) 
        """

        # initialize weights
        for layer in [self.q, self.k, self.v]:
            D_in, D_out = layer.weight.shape
            c = math.sqrt(6 / (D_in + D_out))
            nn.init.uniform_(layer.weight, a=-c, b=c)
        '''
            c1 = math.sqrt(6 / (dim_in + dim_q))
            c2 = math.sqrt(6 / (dim_in + dim_v))

            nn.init.uniform_(self.q.weight, a=-c1, b=c1)
            nn.init.uniform_(self.k.weight, a=-c1, b=c1)
            nn.init.uniform_(self.v.weight, a=-c2, b=c2)
        '''

    def forward(
            self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)
            mask: Tensor of shape (N, K, K)
        Returns:
            y: Tensor of shape (N, K, dim_v)
        """

        self.weights_softmax = (
            None
        )
        y = None

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        y, self.weights_softmax = scaled_dot_product_no_loop_batch(query, key, value, mask)

        return y