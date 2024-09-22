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
    out = []

    words = input_str.split()

    for i, w in enumerate(words):
        if w in spc_tokens:
            out.append(token_dict[w])
        else:
            for ch in w:
                out.append(token_dict[ch])

    return out