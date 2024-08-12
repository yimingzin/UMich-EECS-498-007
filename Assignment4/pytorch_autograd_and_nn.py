import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

to_float = torch.float
to_long = torch.long


def flatten(x, start_dim = 1, end_dim = -1):
    return x.flatten(start_dim, end_dim)

def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params

    x = F.relu(F.conv2d(x, conv_w1, conv_b1, stride=1, padding=2))
    x = F.relu(F.conv2d(x, conv_w2, conv_b2, stride=1, padding=1))
    x = flatten(x)
    scores = F.linear(x, fc_w, fc_b)

    return scores

def initialize_three_layer_conv_part2(dtype = torch.float, device = 'cpu'):
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 32
    channel_2 = 16

    kernel_size_1 = 5
    kernel_size_2 = 3

    conv_w1 = nn.init.kaiming_normal_(torch.empty(channel_1, C, kernel_size_1, kernel_size_1, dtype=dtype, device=device))
    conv_b1 = nn.init.zeros_(torch.empty(channel_1, dtype=dtype, device=device))
    conv_w2 = nn.init.kaiming_normal_(torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, dtype=dtype, device=device))
    conv_b2 = nn.init.zeros_(torch.empty(channel_2, dtype=dtype, device=device))
    fc_w = nn.init.kaiming_normal_(torch.empty(num_classes, channel_2 * H * W, dtype = dtype, device=device))
    fc_b = nn.init.zeros_(torch.empty(num_classes, dtype=dtype, device=device))

    conv_w1.requires_grad = True
    conv_b1.requires_grad = True
    conv_w2.requires_grad = True
    conv_b2.requires_grad = True
    fc_w.requires_grad = True
    fc_b.requires_grad = True

    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    return params


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(channel_2*32*32, num_classes)

        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.kaiming_normal_(self.conv_2.weight)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.conv_1.bias)
        nn.init.zeros_(self.conv_2.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        conv_scores = F.relu(self.conv_2(F.relu(self.conv_1(x))))
        scores = self.fc(flatten(conv_scores))
        return scores


def initialize_three_layer_conv_part3():
    C = 3
    num_classes = 10

    channel_1 = 32
    channel_2 = 16

    learning_rate = 3e-3
    weight_decay = 1e-4

    model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, optimizer

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def initialize_three_layer_conv_part4():
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    pad_size_1 = 2
    kernel_size_2 = 3
    pad_size_2 = 1

    learning_rate = 1e-2
    weight_decay = 1e-4
    momentum = 0.5

    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(C, channel_1, kernel_size_1, stride=1, padding=pad_size_1)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size_2, stride=1, padding=pad_size_2)),
        ('relu2', nn.ReLU()),
        ('flatten', Flatten()),
        ('fc', nn.Linear(channel_2 * H * W, num_classes)),
    ]))

    optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                          weight_decay=weight_decay, momentum=momentum, nesterov=True)

    return model, optimizer
