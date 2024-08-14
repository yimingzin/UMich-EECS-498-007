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


# ---------------------------------------------------------------------------------------------------
# # you should see [64, 10]
# two_layer_fc_test()
# ---------------------------------------------------------------------------------------------------
# # you should see [64, 10]
# three_layer_convnet_test()
# ---------------------------------------------------------------------------------------------------
# # train two layer net
# reset_seed(0)
# learning_rate = 1e-2
# params = initialize_two_layer_fc()
# acc_hist_part2 = train_part2(two_layer_fc, params, learning_rate)
# ---------------------------------------------------------------------------------------------------
# # train three conv net
# reset_seed(0)
# learning_rate = 3e-3
# params = initializer_three_layer_conv_part2()
# acc_hist_part2_conv = train_part2(three_layer_convnet, params, learning_rate)
# ---------------------------------------------------------------------------------------------------
# # you should got [64, 10]
# test_TwoLayerFC()
# ---------------------------------------------------------------------------------------------------
# # you should got [64, 10]
# test_ThreeLayerConvNet()
# ---------------------------------------------------------------------------------------------------
# # train implement with nn.Module TwoLayerFC (part3)
# initialize_two_layer_fc_part3()
# ---------------------------------------------------------------------------------------------------
# # train implement with nn.Module Three-Layer ConvNet (part3)
# reset_seed(0)
# model, optimizer = initializer_three_layer_conv_part3()
# acc_hist_part3, iter_hist_part3 = train_part345(optimizer, model)
# print(acc_hist_part3)
# print(iter_hist_part3)
# ---------------------------------------------------------------------------------------------------
# # train implement with nn.Sequential TwoLayerFC (part4)
# reset_seed(0)
# model, optimizer = initialize_two_layer_part4()
# print('Architecture: ')
# print(model)
# acc_hist_part4, _ = train_part345(optimizer, model)
# ---------------------------------------------------------------------------------------------------
# reset_seed(0)
#
# # YOUR_TURN: Impelement initialize_three_layer_conv_part4
# model, optimizer = initialize_three_layer_conv_part4()
# print('Architecture:')
# print(model) # printing `nn.Module` shows the architecture of the module.
#
# acc_hist_part4, _ = train_part345(optimizer, model)
# print(acc_hist_part4)
# ---------------------------------------------------------------------------------------------------