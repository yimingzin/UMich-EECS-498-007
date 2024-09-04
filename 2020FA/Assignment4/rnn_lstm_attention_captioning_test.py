import math
import os

import torch
import wget
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from a4_helper import *
from eecs598.utils import reset_seed, tensor_to_image, decode_captions, attention_visualizer
from eecs598.grad import rel_error, compute_numeric_gradient
from rnn_lstm_attention_captioning import *
import matplotlib.pyplot as plt
import time


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# data type and device for torch.tensor
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}

# Download and load serialized COCO data from coco.pt
# It contains a dictionary of
# "train_images" - resized training images (112x112)
# "val_images" - resized validation images (112x112)
# "train_captions" - tokenized and numericalized training captions
# "val_captions" - tokenized and numericalized validation captions
# "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

import os
import urllib.request

if os.path.isfile('./datasets/coco.pt'):
    print('COCO data exists')
else:
    print('Downloading COCO dataset')
    url = 'http://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt'
    urllib.request.urlretrieve(url, './datasets/coco.pt')


# load COCO data from coco.pt, loaf_COCO is implemented in a4_helper.py
data_dict = load_COCO(path = './datasets/coco.pt')

num_train = data_dict['train_images'].size(0)
num_val = data_dict['val_images'].size(0)

# declare variables for special tokens
NULL_index = data_dict['vocab']['token_to_idx']['<NULL>']
START_index = data_dict['vocab']['token_to_idx']['<START>']
END_index = data_dict['vocab']['token_to_idx']['<END>']
UNK_index = data_dict['vocab']['token_to_idx']['<UNK>']

# ----------------------------------------------------------------------------------------------------
# # Sample a minibatch and show the reshaped 112x112 images and captions
# batch_size = 3
#
# sample_idx = torch.randint(0, num_train, (batch_size,))
# sample_images = data_dict['train_images'][sample_idx]
# sample_captions = data_dict['train_captions'][sample_idx]
# for i in range(batch_size):
#   plt.imshow(sample_images[i].permute(1, 2, 0))
#   plt.axis('off')
#   caption_str = decode_captions(sample_captions[i], data_dict['vocab']['idx_to_token'])
#   plt.title(caption_str)
#   plt.show()
# ----------------------------------------------------------------------------------------------------
# N, D, H = 3, 10, 4
#
# x = torch.linspace(-0.4, 0.7, steps=N*D, **to_double_cuda).reshape(N, D)
# prev_h = torch.linspace(-0.2, 0.5, steps=N*H, **to_double_cuda).reshape(N, H)
# Wx = torch.linspace(-0.1, 0.9, steps=D*H, **to_double_cuda).reshape(D, H)
# Wh = torch.linspace(-0.3, 0.7, steps=H*H, **to_double_cuda).reshape(H, H)
# b = torch.linspace(-0.2, 0.4, steps=H, **to_double_cuda)
#
# # YOUR_TURN: Impelement rnn_step_forward
# next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
# expected_next_h = torch.tensor([
#   [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
#   [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
#   [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]], **to_double_cuda)
#
# print('next_h error: ', rel_error(expected_next_h, next_h))
# ----------------------------------------------------------------------------------------------------
# reset_seed(0)
# N, D, H = 4, 5, 6
# x = torch.randn(N, D, **to_double_cuda)
# h = torch.randn(N, H, **to_double_cuda)
# Wx = torch.randn(D, H, **to_double_cuda)
# Wh = torch.randn(H, H, **to_double_cuda)
# b = torch.randn(H, **to_double_cuda)
#
# out, cache = rnn_step_forward(x, h, Wx, Wh, b)
#
# dnext_h = torch.randn(*out.shape, **to_double_cuda)
#
# fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fh = lambda h: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]
#
# dx_num = compute_numeric_gradient(fx, x, dnext_h)
# dprev_h_num = compute_numeric_gradient(fh, h, dnext_h)
# dWx_num = compute_numeric_gradient(fWx, Wx, dnext_h)
# dWh_num = compute_numeric_gradient(fWh, Wh, dnext_h)
# db_num = compute_numeric_gradient(fb, b, dnext_h)
#
# # YOUR_TURN: Impelement rnn_step_backward
# dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
#
# print('dx error: ', rel_error(dx_num, dx))
# print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
# print('dWx error: ', rel_error(dWx_num, dWx))
# print('dWh error: ', rel_error(dWh_num, dWh))
# print('db error: ', rel_error(db_num, db))
# ----------------------------------------------------------------------------------------------------
N, T, D, H = 2, 3, 4, 5

x = torch.linspace(-0.1, 0.3, steps=N*T*D, **to_double_cuda).reshape(N, T, D)
h0 = torch.linspace(-0.3, 0.1, steps=N*H, **to_double_cuda).reshape(N, H)
Wx = torch.linspace(-0.2, 0.4, steps=D*H, **to_double_cuda).reshape(D, H)
Wh = torch.linspace(-0.4, 0.1, steps=H*H, **to_double_cuda).reshape(H, H)
b = torch.linspace(-0.7, 0.1, steps=H, **to_double_cuda)

# YOUR_TURN: Impelement rnn_forward
h, _ = rnn_forward(x, h0, Wx, Wh, b)
expected_h = torch.tensor([
  [
    [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
    [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
    [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
  ],
  [
    [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
    [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
    [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]], **to_double_cuda)
print('h error: ', rel_error(expected_h, h))

reset_seed(0)

N, D, T, H = 2, 3, 10, 5

x = torch.randn(N, T, D, **to_double_cuda)
h0 = torch.randn(N, H, **to_double_cuda)
Wx = torch.randn(D, H, **to_double_cuda)
Wh = torch.randn(H, H, **to_double_cuda)
b = torch.randn(H, **to_double_cuda)

out, cache = rnn_forward(x, h0, Wx, Wh, b)

dout = torch.randn(*out.shape, **to_double_cuda)

# YOUR_TURN: Impelement rnn_backward
dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

dx_num = compute_numeric_gradient(fx, x, dout)
dh0_num = compute_numeric_gradient(fh0, h0, dout)
dWx_num = compute_numeric_gradient(fWx, Wx, dout)
dWh_num = compute_numeric_gradient(fWh, Wh, dout)
db_num = compute_numeric_gradient(fb, b, dout)

print('dx error: ', rel_error(dx_num, dx))
print('dh0 error: ', rel_error(dh0_num, dh0))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))

reset_seed(0)

N, D, T, H = 2, 3, 10, 5

# set requires_grad=True
x = torch.randn(N, T, D, **to_double_cuda, requires_grad=True)
h0 = torch.randn(N, H, **to_double_cuda, requires_grad=True)
Wx = torch.randn(D, H, **to_double_cuda, requires_grad=True)
Wh = torch.randn(H, H, **to_double_cuda, requires_grad=True)
b = torch.randn(H, **to_double_cuda, requires_grad=True)

out, cache = rnn_forward(x, h0, Wx, Wh, b)

dout = torch.randn(*out.shape, **to_double_cuda)

# manual backward
with torch.no_grad():
  dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

# backward with autograd
out.backward(dout) # the magic happens here!
dx_auto, dh0_auto, dWx_auto, dWh_auto, db_auto = \
  x.grad, h0.grad, Wx.grad, Wh.grad, b.grad

print('dx error: ', rel_error(dx_auto, dx))
print('dh0 error: ', rel_error(dh0_auto, dh0))
print('dWx error: ', rel_error(dWx_auto, dWx))
print('dWh error: ', rel_error(dWh_auto, dWh))
print('db error: ', rel_error(db_auto, db))