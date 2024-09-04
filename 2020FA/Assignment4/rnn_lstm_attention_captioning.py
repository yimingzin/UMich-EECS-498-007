import torch
import math
import torch.nn as nn
from a4_helper import *
from torch.nn.parameter import Parameter


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    :param x: 输入数据，形状为（N，D），其中 N 是批量大小，D 是输入数据的维度。
    :param prev_h: 上一个时间步的隐藏状态，形状为（N，H），其中H是隐层状态的维度
    :param Wx: 输入到隐藏层的权重矩阵，形状为（D，H）
    :param Wh: 隐藏层到隐藏层的权重矩阵，形状为（H，H）
    :param b: 偏置项，形状为（H，）
    :return: next_h: 当前时间步的隐藏状态，作为下一个时间步的prev_h, 形状为(N, H)
    """

    next_h = torch.tanh(torch.mm(x, Wx) + torch.mm(prev_h, Wh) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    x, prev_h, Wx, Wh, b, next_h = cache

    dtanh = dnext_h * (1 - next_h ** 2)

    dx = torch.mm(dtanh, Wx.t())  # (N, D) = (N, H) @ (H, D)
    dprev_h = torch.mm(dtanh, Wh.t())  # (N, H) = (N, H) @ (H, H)
    dWx = torch.mm(x.t(), dtanh)    # (D, H) = (D, N) @ (N, H)
    dWh = torch.mm(prev_h.t(), dtanh)   # (H, H) = (H, N) @ (N, H)
    db = dtanh.sum(dim=0)

    return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    H = h0.shape[1]

    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    cache = []

    prev_h = h0
    for t in range(T):
        next_h, step_cache = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(step_cache)
        prev_h = next_h

    return h, cache

def rnn_backward(dh, cache):
    # 从dh中获取形状信息
    N, T, H = dh.shape
    D = cache[0][0].shape[1]

    # 初始化梯度
    dx = torch.zeros((N, T, D), dtype=dh.dtype, device=dh.device)
    dWx = torch.zeros((D, H), dtype=dh.dtype, device=dh.device)
    dWh = torch.zeros((H, H), dtype=dh.dtype, device=dh.device)
    db = torch.zeros((H,), dtype=dh.dtype, device=dh.device)
    dh0 = torch.zeros((N, H), dtype=dh.dtype, device=dh.device)

    # 初始化前一个时间步的隐藏状态梯度
    dprev_h = torch.zeros((N, H), dtype=dh.dtype, device=dh.device)

    # 反向遍历所有时间步
    for t in reversed(range(T)):
        # 将来自时间步 t 的 dh 和传递下来的 dprev_h 累加
        dnext_h = dh[:, t, :] + dprev_h

        # 使用 rnn_step_backward 计算每个时间步的梯度
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])

        # 累加计算得到的梯度
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    # 最后的 dprev_h 就是 dh0
    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db
