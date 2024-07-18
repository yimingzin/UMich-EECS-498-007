import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional

"""
    支持向量机损失函数朴素实现
"""
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
#   外层循环遍历num_train
    for i in range(num_train):
        #计算当前num_train在所有种类的得分
        scores = torch.mv(W.t(), X[i])
        #正确标签得分
        correct_scores = scores[y[i]]
        for j in range(num_class):
            #计算损失函数只计算错误标签
            if j == y[i]:
                continue
            # SVM Loss - 计算间隔，当前num_train的其他类别得分 - 正确类别得分, 看间隔是否大于1
            margin = scores[j] - correct_scores + 1
            # 更新损失和梯度, 损失即直接加间隔, 梯度求导后其他类别相加，正确类别相减
            if margin > 0 :
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    #正则化
    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW

def softmax_loss_naive(
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
        scores = torch.mv(W.t(), X[i])
        #上面同svm实现，softmax计算概率，需要保证数字稳定性即减去每个num_train中最大的得分
        scores_stable = scores - scores.max()
        correct_scores = scores_stable[y[i]]
        sum_exp = torch.sum(torch.exp(scores_stable))
        #softmax的损失函数 = -log(e^(正确类别得分) / e^(所有类别得分)求和 )
        loss += torch.log(sum_exp) - correct_scores
        #梯度计算，对于正确类别需要-1，其他类别直接计算即可 - 核心是e^(类别得分) / e^(所有类别得分)求和，这里把正确类别和错误类别分开讨论
        for j in range(num_class):
            if j == y[i]:
                dW[:, j] += (torch.exp(scores_stable[j])/ sum_exp - 1) * X[i]
            else:
                dW[:, j] += torch.exp(scores_stable[j]) / sum_exp * X[i]

    #正则化
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

    #计算得分 shape = (num_train, num_class)
    scores = X.mm(W)
    #计算正确得分，这里把本来为1行num_train列化为num_train行1列为了方便下面计算
    correct_scores = scores[range(num_train), y].reshape(-1, 1)
    zeros = torch.zeros_like(scores)
    #计算间隔，scores.shape = (num_train, num_class), 而正确类别得分是(num_train, 1),相减时通过广播机制自动扩充至(num_train, num_class)
    #把当前num_train的正确分数复制num_class份再相减(横向复制), maximum和0矩阵逐个元素比较，大者胜出
    margin = torch.maximum(scores - correct_scores + 1, zeros)
    #除去正确类别影响，如果不除去的话对于正确类别scores - correct_scores = 0，结果为1会影响计算
    margin[range(num_train), y] = 0
    loss = margin.sum() / num_train + reg * torch.sum(W * W)

    #把间隔大于0标记为1
    margin[margin > 0] = 1
    margin[margin < 0] = 0
    #当前margin的正确类别抵消了多少个错误类比
    margin[range(num_train), y] = -torch.sum(margin, dim=1)

    #计算梯度得到dW.shape = (dim, Class)
    dW = torch.mm(X.T, margin) / num_train + 2 * reg * W

    return loss, dW

def softmax_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    scores = torch.mm(X, W)
    #通过keepdim = True 让其保持 m行 x 1列的形式，下softmax同, (max会返回值和索引，只需要值)
    scores_stable = scores - scores.max(dim=1, keepdim=True).values
    correct_scores = scores_stable[range(num_train), y]
    # e ^ (scores)
    exp = torch.exp(scores_stable)
    softmax = exp / exp.sum(dim=1, keepdim=True)


    loss = -correct_scores + torch.log(torch.sum(exp, dim=1))
    loss = loss.sum()
    loss = loss / num_train + reg * torch.sum(W * W)

    #计算梯度，把正确类别置为-1，和核心softmax相加 核心是e ^ (类别得分) / e ^ (所有类别得分)求和，
    correct_matrix = torch.zeros_like(scores)
    correct_matrix[range(num_train), y] = -1
    dW += torch.mm(X.t(), softmax + correct_matrix)
    dW = dW / num_train + 2 * reg * W

    return loss, dW


