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

#----------------------------------------------------------------------------------------------------------
# def two_layer_fc(x, params):
#     """
#     A fully-connected neural networks; the architecture is:
#     NN is fully connected -> ReLU -> fully connected layer.
#     Note that this function only defines the forward pass;
#     PyTorch will take care of the backward pass for us.
#
#     The input to the network will be a minibatch of data, of shape
#     (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
#     and the output layer will produce scores for C classes.
#
#     Inputs:
#     - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
#       input data.
#     - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
#       w1 has shape (H, D) and w2 has shape (C, H).
#
#     Returns:
#     - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
#       the input data x.
#     """
#     # first we flatten the image
#     # note: start_dim & end_dim
#     x = flatten(x, start_dim=1)  # shape: [batch_size, C x H x W]
#
#     w1, b1, w2, b2 = params
#
#     # Forward pass: compute predicted y using operations on Tensors. Since w1 and
#     # w2 have requires_grad=True, operations involving these Tensors will cause
#     # PyTorch to build a computational graph, allowing automatic computation of
#     # gradients. Since we are no longer implementing the backward pass by hand we
#     # don't need to keep references to intermediate values.
#     # Note that F.linear(x, w, b) is equivalent to x.mm(w.t()) + b
#     # For ReLU, you can also use `.clamp(min=0)`, equivalent to `F.relu()`
#     x = F.relu(F.linear(x, w1, b1))
#     x = F.linear(x, w2, b2)
#     return x
#
#
# def two_layer_fc_test():
#     hidden_layer_size = 42
#     x = torch.zeros((64, 3, 16, 16), dtype=to_float)  # minibatch size 64, feature dimension 3*16*16
#     w1 = torch.zeros((hidden_layer_size, 3 * 16 * 16), dtype=to_float)
#     b1 = torch.zeros((hidden_layer_size,), dtype=to_float)
#     w2 = torch.zeros((10, hidden_layer_size), dtype=to_float)
#     b2 = torch.zeros((10,), dtype=to_float)
#     scores = two_layer_fc(x, [w1, b1, w2, b2])
#     print('Output size:', list(scores.size()))  # you should see [64, 10]
#
#
# two_layer_fc_test()
#----------------------------------------------------------------------------------------------------------
# #test three_layer_convnet
# def three_layer_convnet_test():
#   x = torch.zeros((64, 3, 32, 32), dtype=to_float)  # minibatch size 64, image size [3, 32, 32]
#
#   conv_w1 = torch.zeros((6, 3, 5, 5), dtype=to_float)  # [out_channel, in_channel, kernel_H, kernel_W]
#   conv_b1 = torch.zeros((6,))  # out_channel
#   conv_w2 = torch.zeros((9, 6, 3, 3), dtype=to_float)  # [out_channel, in_channel, kernel_H, kernel_W]
#   conv_b2 = torch.zeros((9,))  # out_channel
#
#   # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
#   fc_w = torch.zeros((10, 9 * 32 * 32))
#   fc_b = torch.zeros(10)
#
#   # YOUR_TURN: Impelement the three_layer_convnet function
#   scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
#   print('Output size:', list(scores.size()))  # you should see [64, 10]
# three_layer_convnet_test()
#----------------------------------------------------------------------------------------------------------
reset_seed(0)

def check_accuracy_part2(loader, model_fn, params):
    """
  Check the accuracy of a classification model.

  Inputs:
  - loader: A DataLoader for the data split we want to check
  - model_fn: A function that performs the forward pass of the model,
    with the signature scores = model_fn(x, params)
  - params: List of PyTorch Tensors giving parameters of the model

  Returns: Nothing, but prints the accuracy of the model
  """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cuda', dtype=to_float)  # move to device, e.g. GPU
            y = y.to(device='cuda', dtype=to_long)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
        return acc


def train_part2(model_fn, params, learning_rate):
    """
  Train a model on CIFAR-10.

  Inputs:
  - model_fn: A Python function that performs the forward pass of the model.
    It should have the signature scores = model_fn(x, params) where x is a
    PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
    model weights, and scores is a PyTorch Tensor of shape (N, C) giving
    scores for the elements in x.
  - params: List of PyTorch Tensors giving weights for the model
  - learning_rate: Python scalar giving the learning rate to use for SGD

  Returns: Nothing
  """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device='cuda', dtype=to_float)
        y = y.to(device='cuda', dtype=to_long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                if w.requires_grad:
                    w -= learning_rate * w.grad

                    # Manually zero the gradients after running the backward pass
                    w.grad.zero_()

        if t % 100 == 0 or t == len(loader_train) - 1:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            acc = check_accuracy_part2(loader_val, model_fn, params)
    return acc


reset_seed(0)

C, H, W = 3, 32, 32
num_classes = 10

hidden_layer_size = 4000
learning_rate = 1e-2


#----------------------------------------------------------------------------------------------------------
# def two_layer_fc(x, params):
#     x = flatten(x)  # shape: [batch_size, C x H x W]
#
#     w1, b1, w2, b2 = params
#
#     x = F.relu(F.linear(x, w1, b1))
#     x = F.linear(x, w2, b2)
#     return x
#
#
# w1 = nn.init.kaiming_normal_(torch.empty(hidden_layer_size, C * H * W, dtype=to_float, device='cuda'))
# w1.requires_grad = True
# b1 = nn.init.zeros_(torch.empty(hidden_layer_size, dtype=to_float, device='cuda'))
# b1.requires_grad = True
# w2 = nn.init.kaiming_normal_(torch.empty(num_classes, hidden_layer_size, dtype=to_float, device='cuda'))
# w2.requires_grad = True
# b2 = nn.init.zeros_(torch.empty(num_classes, dtype=to_float, device='cuda'))
# b2.requires_grad = True
#
# _ = train_part2(two_layer_fc, [w1, b1, w2, b2], learning_rate)

#----------------------------------------------------------------------------------------------------------
reset_seed(0)
learning_rate = 3e-3
# YOUR_TURN: Impelement the initialize_three_layer_conv_part2 function
params = initialize_three_layer_conv_part2(dtype=to_float, device='cuda')
acc_hist_part2 = train_part2(three_layer_convnet, params, learning_rate)

