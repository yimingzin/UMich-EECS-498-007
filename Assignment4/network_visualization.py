import torch
import matplotlib.pyplot as plt
from PIL import Image
from a4_helper import *
import torch.nn as nn
import torch.nn.functional as F


def compute_saliency_maps(X, y, model):
    model.eval()
    X.requires_grad_()

    saliency = None
    #计算整体分数
    scores = model(X)
    #通过gather获取正确分数, 先把y索引转为列向量后沿着dim=1从scores中得出
    correct_scores = torch.gather(scores, 1, y.view(-1, 1))
    #求正确分数对像素X的梯度, 梯度表明了哪些像素对最终得分影响最大，我们求显著图就是求这些
    loss = torch.sum(correct_scores)
    loss.backward()
    grads = X.grad.data
    #对梯度取绝对值和在RGB三个通道上的最大值, 从(N, 3, H, W) => (N, H, W)
    saliency = torch.abs(grads)
    saliency = torch.max(saliency, dim=1).values

    return saliency


