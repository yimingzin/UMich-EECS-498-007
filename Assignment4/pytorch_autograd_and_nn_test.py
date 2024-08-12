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
#     x = torch.zeros((64, 3, 32, 32), dtype=to_float, device='cuda')
#
#     params = initialize_three_layer_conv_part2(dtype=to_float, device='cuda')
#     scores = three_layer_convnet(x, params)
#     print('Output size: ', list(scores.size()))
#
#
# three_layer_convnet_test()


# ------------------------------------------------------------------------------------------------------

def check_accuracy_part2(loader, model_fn, params):
    split = 'val' if loader.dataset.train else 'test'
    print('now checking Accuracy on %s set' % split)
    num_correct, num_samples = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dtype=to_float, device='cuda')
            y = y.to(dtype=to_long, device='cuda')
            scores = model_fn(x, params)
            preds = torch.argmax(scores, dim=1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(dim=0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct, accuracy = %.2f%%' % (num_correct, num_samples, acc * 100))
        return acc


def train_part2(model_fn, params, learning_rate):
    for t, (x, y) in enumerate(loader_train):
        x = x.to(dtype=to_float, device='cuda')
        y = y.to(dtype=to_long, device='cuda')

        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)
        loss.backward()

        with torch.no_grad():
            for w in params:
                if w.requires_grad:
                    w -= learning_rate * w.grad
                    w.grad.zero_()

        if t % 100 == 0 or t == len(loader_train) - 1:
            print('Iteration %d , loss = %.4f ' % (t, loss.item()))
            acc = check_accuracy_part2(loader_val, model_fn, params)

    return acc


# ------------------------------------------------------------------------------------------------------
# reset_seed(0)
# learning_rate = 3e-3
# params = initialize_three_layer_conv_part2(dtype = to_float, device = 'cuda')
# acc_hist_part2 = train_part2(three_layer_convnet, params, learning_rate)
# ------------------------------------------------------------------------------------------------------
class TwoLayerFC(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    # assign layer objects to class attributes
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)
    nn.init.kaiming_normal_(self.fc1.weight)
    nn.init.kaiming_normal_(self.fc2.weight)
    nn.init.zeros_(self.fc1.bias)
    nn.init.zeros_(self.fc2.bias)

  def forward(self, x):
    # 在 PyTorch 的 nn.Module 类中，__call__ 方法专门设计为调用 forward 方法
    # 当调用一个模型对象（如 model(x)）时，实际上是调用了模型对象的 __call__ 方法
    x = flatten(x)
    scores = self.fc2(F.relu(self.fc1(x)))
    return scores

def test_TwoLayerFC():
  input_size = 3*16*16
  x = torch.zeros((64, input_size), dtype=to_float)  # minibatch size 64, feature dimension 3*16*16
  model = TwoLayerFC(input_size, 42, 10)
  scores = model(x)
  print('Architecture:')
  print(model) # printing `nn.Module` shows the architecture of the module.
  print('Output size:', list(scores.size()))  # you should see [64, 10]
test_TwoLayerFC()

# ------------------------------------------------------------------------------------------------------
# def test_ThreeLayerConvNet():
#     x = torch.zeros((64, 3, 32, 32), dtype=to_float)
#     model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
#     scores = model(x)
#     print('Output size: ', list(scores.size()))
#
# test_ThreeLayerConvNet()
# ------------------------------------------------------------------------------------------------------
def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Now check accuracy on val set')
    else:
        print('Now check accuracy on test set')
    num_correct, num_samples = 0, 0

    # model.eval() 在计算验证集或测试集的准确率之前被调用，目的是确保模型的行为与训练时不同
    # 如果模型中有 dropout 层，它们将停止随机丢弃神经元的输出
    # 批归一化层将使用在训练时记录的均值和方差，而不是当前批次的数据统计信息
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dtype=to_float, device='cuda')
            y = y.to(dtype=to_long, device='cuda')
            scores = model(x)
            preds = torch.argmax(scores, dim=1)

            num_correct += (y == preds).sum()
            num_samples += preds.size(dim=0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)%%' % (num_correct, num_samples, acc * 100))

    return acc


def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    if epoch in schedule:
        for param_groups in optimizer.param_groups:
            print('learning rate decay from {} to {}'.format(param_groups['lr'], param_groups['lr'] * lrd))
            param_groups['lr'] *= lrd

def train_part345(model, optimizer, epoch = 1, schedule = [], learning_rate_decay = .1, verbose = True):
    model = model.to(device = 'cuda')
    num_iters = epoch * len(loader_train)
    print_every = 100
    if verbose:
        num_prints = num_iters // print_every + 1
    else:
        num_prints = epoch
    acc_history = torch.zeros(num_prints, dtype = to_float)
    iter_history = torch.zeros(num_prints, dtype = to_long)

    for e in range(epoch):
        adjust_learning_rate(optimizer, learning_rate_decay, e, schedule)

        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(dtype = to_float, device = 'cuda')
            y = y.to(dtype = to_long, device = 'cuda')
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tt = t + e * len(loader_train)
            if verbose and (tt % print_every == 0 or (e == epoch - 1 and t == len(loader_train) - 1)):
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc_history[tt // print_every] = check_accuracy_part34(loader_val, model)
                iter_history[tt // print_every] = tt
                print()
            elif not verbose and (t == len(loader_train) - 1):
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[e] = acc
                iter_history[e] = tt
                print()
    return acc_history, iter_history


# ------------------------------------------------------------------------------------------------------
# reset_seed(0)
# C, H, W = 3, 32, 32
# hidden_layer_size = 4000
# num_classes = 10
#
# learning_rate = 1e-2
# weight_decay = 1e-4
# model = TwoLayerFC(C*H*W, hidden_layer_size, num_classes)
# optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
#
# _ = train_part345(model, optimizer)
# ------------------------------------------------------------------------------------------------------
# reset_seed(0)
# model, optimizer = initialize_three_layer_conv_part3()
# acc_hist_part3, _ = train_part345(model, optimizer)
# ------------------------------------------------------------------------------------------------------

# reset_seed(0)
# C, H, W = 3, 32, 32
# num_classes = 10
# hidden_layer_size = 4000
# learning_rate = 1e-2
# weight_decay = 1e-4
# momentum = 0.5
#
# model = nn.Sequential(OrderedDict([
#     ('flatten', Flatten()),
#     ('fc1', nn.Linear(C*H*W, hidden_layer_size)),
#     ('relu1', nn.ReLU()),
#     ('fc2', nn.Linear(hidden_layer_size, num_classes)),
# ]))
#
# print('Architecture: ')
# print(model)
#
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)
# _ = train_part345(model, optimizer)

# ------------------------------------------------------------------------------------------------------
reset_seed(0)
model, optimizer = initialize_three_layer_conv_part4()
print('Architecture: ')
print(model)
acc_hist_part4, _ = train_part345(model, optimizer)
