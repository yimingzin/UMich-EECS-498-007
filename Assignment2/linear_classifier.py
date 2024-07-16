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

            if margin > 0 :
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
