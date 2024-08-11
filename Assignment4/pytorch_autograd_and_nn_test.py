import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from eecs598.utils import reset_seed
from collections import OrderedDict
from pytorch_autograd_and_nn import *
from a4_helper import *

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

to_float = torch.float
to_long = torch.long

loader_train, loader_val, loader_test = load_CIFAR(path='./datasets/')


# ------------------------------------------------------------------------------------------------------
# def three_layer_convnet_test():
#     x = torch.zeros((64, 3, 32, 32), dtype=to_float)
#
#     # [out_channel(num_filters), in_channel, kernel_H, kernel_W]
#     conv_w1 = torch.zeros((6, 3, 5, 5), dtype=to_float)
#     # [out_channel]
#     conv_b1 = torch.zeros(6, dtype = to_float)
#     conv_w2 = torch.zeros((9, 6, 3, 3), dtype=to_float)
#     conv_b2 = torch.zeros(9, dtype=to_float)
#
#     #[num_classes, num_filters * H_prime * W_prime]
#     fc_w = torch.zeros((10, 9 * 32 * 32), dtype=to_float)
#     fc_b = torch.zeros(10, dtype=to_float)
#
#     params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
#     scores = three_layer_convnet(x, params)
#     print('Output size: ', list(scores.size()))
#
# three_layer_convnet_test()
# ------------------------------------------------------------------------------------------------------
def check_accuracy_part2(loader, model_fn, params):
    """
    在不计算梯度(节省内存和资源)的情况下，评估模型在验证集或测试集上的准确率
    :param loader: PyTorch 的 DataLoader 对象，包含要检查的输入数据集和对应的标签。
    :param model_fn: 用于执行模型的前向传播，返回分类分数
    :param params:   模型参数列表，通常有权重和偏置
    :return: acc     准确率
    """
    # loader.dataset.train返回一个布尔值，用于指示数据集是否为训练集,不过我们通常不在训练时候检查准确率，而是使用验证集和测试集
    split = 'val' if loader.dataset.train else split = 'test'
    print('Now checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
