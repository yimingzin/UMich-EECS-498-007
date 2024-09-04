import torch
import math
import torch.nn as nn
from a4_helper import *
from torch.nn.parameter import Parameter


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h = torch.tanh(torch.mm(x, Wx) + torch.mm(prev_h, Wh) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    x, prev_h, Wx, Wh, b, next_h = cache
    dtanh = dnext_h * (1 - next_h ** 2)
    dx = torch.mm(dtanh, Wx.t())
    dprev_h = torch.mm(dtanh, Wh.t())
    dWx = torch.mm(x.t(), dtanh)
    dWh = torch.mm(prev_h.t(), dtanh)
    db = torch.sum(dtanh, dim=0)

    return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    N, H = h0.shape

    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    cache = []
    prev_h = h0

    for t in range(T):
        next_h, cache_step = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(cache_step)
        prev_h = next_h

    return h, cache

def rnn_backward(dh, cache):
    N, T, H = dh.shape
    # cache = [(x[:, t1, :], prev_h1, Wx, Wh, b, next_h1), (x[:, t2, :], prev_h2, Wx, Wh, b, next_h2), ...]
    # x[:, t, :] 时间步 t 的输入，形状为(N, D)
    D = cache[0][0].shape[1]

    # dx = torch.zeros_like(cache[0][0], dtype=dh.dtype, device=dh.device), cache[0][0] 是(N, D)形状而不是(N, T, D)
    dx = torch.zeros((N, T, D), dtype=dh.dtype, device=dh.device)
    dprev_h = torch.zeros_like(cache[0][1], dtype=dh.dtype, device=dh.device)
    dWx = torch.zeros_like(cache[0][2], dtype=dh.dtype, device=dh.device)
    dWh = torch.zeros_like(cache[0][3], dtype=dh.dtype, device=dh.device)
    db = torch.zeros_like(cache[0][4], dtype=dh.dtype, device=dh.device)

    # 在实际中不需要把 dh0 单独初始化为 0，会在反向传播过程中被赋值为最后一个 dprev_h 的值, 这里是用来检验最后的结果正确与否的.
    dh0 = torch.zeros((N, H), dtype=dh.dtype, device=dh.device)

    # for t in range(T-1, -1, -1):
    for t in reversed(range(T)):
        # 当前时间步t的隐藏状态的总梯度 = 直接损失的梯度和通过时间序列传递回来的梯度
        dnext_h = dh[:, t, :] + dprev_h

        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])

        # 当前时间步 t 输入梯度dx_t保存到dx中的对应位置，同时累加梯度
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    # 在反向传播结束时最后一次计算得到的dprev_h就是初始隐藏状态h0的梯度, 用来调整初始隐藏状态，确保更好适应序列数据
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db






