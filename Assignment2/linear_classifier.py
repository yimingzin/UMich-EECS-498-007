import torch
import random
import statistics
from abc import abstractclassmethod
from typing import Dict, List, Callable, Optional

def svm_loss_naive(
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        reg: float
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_scores = scores[y[i]]
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j] - correct_scores + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW

def svm_loss_vectorized(
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        reg: float,
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    scores = X.mm(W)
    correct_scores = scores[range(num_train), y].reshape(-1, 1)
    zeros = torch.zeros_like(scores)

    margin = torch.maximum(scores - correct_scores + 1, zeros)
    margin[range(num_train), y] = 0

    loss = margin.sum() / num_train + reg * torch.sum(W * W)

    margin[margin > 0] = 1
    margin[margin < 0] = 0

    margin[range(num_train), y] = -torch.sum(margin, dim=1)

    temp = X.t()
    dW = temp.mm(scores) / num_train + 2 * reg * W

    return loss, dW
