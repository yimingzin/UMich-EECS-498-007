import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

to_float = torch.float
to_long = torch.long


def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)


def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    x = F.relu(F.conv2d(x, conv_w1, conv_b1, stride=1, padding=2))
    x = F.relu(F.conv2d(x, conv_w2, conv_b2, stride=1, padding=1))
    x = flatten(x)
    scores = F.linear(x, fc_w, fc_b)

    return scores


def initialize_three_layer_conv_part2(dtype=torch.float, device = 'cpu'):

    # input / output dimenssions
    C, H, W = 3, 32, 32
    num_classes = 10

    # hidden layer channel and kernel size
    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    kernel_size_2 = 3

    conv_w1 = nn.init.kaiming_normal_(torch.empty(channel_1, C, kernel_size_1, kernel_size_1, dtype=dtype, device=device))
    conv_b1 = nn.init.zeros_(torch.empty(channel_1, dtype=dtype, device=device))

    conv_w2 = nn.init.kaiming_normal_(torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, dtype=dtype, device=device))
    conv_b2 = nn.init.zeros_(torch.empty(channel_2, dtype=dtype, device=device))
    # 传播过程中使用了 ReLU 或其变体进行非线性激活，那么后面的权重最好都使用 Kaiming 初始化
    fc_w = nn.init.kaiming_normal_(torch.empty(num_classes, channel_2 * H * W, dtype=dtype, device=device))
    fc_b = nn.init.zeros_(torch.empty(num_classes, dtype=dtype, device=device))

    conv_w1.requires_grad = True
    conv_b1.requires_grad = True
    conv_w2.requires_grad = True
    conv_b2.requires_grad = True
    fc_w.requires_grad = True
    fc_b.requires_grad = True

    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    return params
