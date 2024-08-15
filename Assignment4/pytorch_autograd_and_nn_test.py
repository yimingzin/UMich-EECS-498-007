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
# data = torch.zeros(2, 3, 5, 6)
# # YOUR_TURN: Implement PlainBlock.__init__
# model = PlainBlock(3, 10)
# if list(model(data).shape) == [2, 10, 5, 6]:
#   print('The output of PlainBlock without downsampling has a *correct* dimension!')
# else:
#   print('The output of PlainBlock without downsampling has an *incorrect* dimension! expected:', [2, 10, 5, 6], 'got:', list(model(data).shape))
#
# data = torch.zeros(2, 3, 5, 6)
# # YOUR_TURN: Impelement PlainBlock.__init__
# model = PlainBlock(3, 10, downsample=True)
# if list(model(data).shape) == [2, 10, 3, 3]:
#   print('The output of PlainBlock with downsampling has a *correct* dimension!')
# else:
#   print('The output of PlainBlock with downsampling has an *incorrect* dimension! expected:', [2, 10, 3, 3], 'got:', list(model(data).shape))
# ---------------------------------------------------------------------------------------------------
# data = torch.zeros(2, 3, 5, 6)
# # YOUR_TURN: Impelement ResidualBlock.__init__
# model = ResidualBlock(3, 10)
# if list(model(data).shape) == [2, 10, 5, 6]:
#   print('The output of ResidualBlock without downsampling has a *correct* dimension!')
# else:
#   print('The output of ResidualBlock without downsampling has an *incorrect* dimension! expected:', [2, 10, 5, 6], 'got:', list(model(data).shape))
#
# data = torch.zeros(2, 3, 5, 6)
# # YOUR_TURN: Impelement ResidualBlock.__init__
# model = ResidualBlock(3, 10, downsample=True)
# if list(model(data).shape) == [2, 10, 3, 3]:
#   print('The output of ResidualBlock with downsampling has a *correct* dimension!')
# else:
#   print('The output of ResidualBlock with downsampling has an *incorrect* dimension! expected:', [2, 10, 3, 3], 'got:', list(model(data).shape))
# ---------------------------------------------------------------------------------------------------
# print('Plain block stage:')
# print(ResNetStage(3, 4, 2, block=PlainBlock))
# print('Residual block stage:')
# print(ResNetStage(3, 4, 2, block=ResidualBlock))
# ---------------------------------------------------------------------------------------------------
# data = torch.zeros(2, 3, 5, 6)
# model = ResNetStem(3, 10)
# if list(model(data).shape) == [2, 10, 5, 6]:
#   print('The output of ResNetStem without downsampling has a *correct* dimension!')
# else:
#   print('The output of ResNetStem without downsampling has an *incorrect* dimension! expected:', [2, 10, 5, 6], 'got:', list(model(data).shape))
# ---------------------------------------------------------------------------------------------------
# example of specifications
networks = {
  'plain32': {
    'block': PlainBlock,
    'stage_args': [
      (8, 8, 5, False),
      (8, 16, 5, True),
      (16, 32, 5, True),
    ]
  },
  'resnet32': {
    'block': ResidualBlock,
    'stage_args': [
      (8, 8, 5, False),
      (8, 16, 5, True),
      (16, 32, 5, True),
    ]
  },
}

def get_resnet(name):
  # YOUR_TURN: Impelement ResNet.__init__ and ResNet.forward
  return ResNet(**networks[name])

names = ['plain32', 'resnet32']
acc_history_dict = {}
iter_history_dict = {}
for name in names:
  reset_seed(0)
  print(name, '\n')
  model = get_resnet(name)
#   init_module(model)

  optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=.9, weight_decay=1e-4)

  acc_history, iter_history = train_part345(optimizer, model, epoch=10, schedule=[6, 8], verbose=False)
  acc_history_dict[name] = acc_history
  iter_history_dict[name] = iter_history

plt.title('Val accuracies')
for name in names:
  plt.plot(iter_history_dict[name], acc_history_dict[name], '-o')
plt.legend(names, loc='upper left')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.gcf().set_size_inches(9, 4)
plt.show()