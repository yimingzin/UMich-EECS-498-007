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
    scores = None

    pad_k1 = 2
    pad_k2 = 1

    scores = F.relu(F.conv2d(x, conv_w1, conv_b1, stride=1, padding=pad_k1))
    scores = F.relu(F.conv2d(scores, conv_w2, conv_b2, stride=1, padding=pad_k2))
    #fc layer
    scores = flatten(scores)
    scores = F.linear(scores, fc_w, fc_b)

    return scores




def initialize_three_layer_conv_part2(dtype=torch.float, device = 'cpu'):
    #input / Output dim
    C, H, W = 3, 32, 32
    num_classes = 10

    #hidden layer channel and kernel sizes
    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    kernel_size_2 = 3

    #Initialize the weights
    conv_w1 = None
    conv_b1 = None
    conv_w2 = None
    conv_b2 = None
    fc_w = None
    fc_b = None

    #normalize the weights
    conv_w1 = torch.nn.init.kaiming_normal_(torch.empty(channel_1, C, kernel_size_1, kernel_size_1, dtype=dtype, device=device))
    conv_b1 = torch.nn.init.zeros_(torch.empty(channel_1, dtype=dtype, device=device))
    conv_w2 = torch.nn.init.kaiming_normal_(torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, dtype=dtype, device=device))
    conv_b2 = torch.nn.init.zeros_(torch.empty(channel_2, dtype=dtype, device=device))
    fc_w = torch.nn.init.kaiming_normal_(torch.empty(num_classes, channel_2 * H * W, dtype=dtype, device=device))
    fc_b = torch.nn.init.zeros_(torch.empty(num_classes, dtype=dtype, device=device))

    conv_w1.requires_grad = True
    conv_b1.requires_grad = True
    conv_w2.requires_grad = True
    conv_b2.requires_grad = True
    fc_w.requires_grad = True
    fc_b.requires_grad = True

    return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]




