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
) -> Tensor:

    K, M = query.shape
    K_k, M = key.shape

    QK = torch.zeros((K, K_k), dtype=query.dtype, device=query.device)

    QK = torch.mm(query, key.transpose(1, 0)) / math.sqrt(M)
    QK = torch.softmax(QK, dim=-1)

    out = torch.mm(QK, value)

    return out

def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

    N, K, M = query.shape
    N, K_k, M = key.shape

    QK = torch.zeros((N, K, K_k), dtype=query.dtype, device=query.device)

    QK = torch.bmm(query, key.permute(0, 2, 1))

    QK = QK / math.sqrt(M)
    QK = torch.softmax(QK, dim=-1)

    out = torch.bmm(QK, value)

    return out
