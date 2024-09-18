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
# N, T, D, H = 2, 3, 4, 5
#
# x = torch.linspace(-0.1, 0.3, steps=N*T*D, **to_double_cuda).reshape(N, T, D)
# h0 = torch.linspace(-0.3, 0.1, steps=N*H, **to_double_cuda).reshape(N, H)
# Wx = torch.linspace(-0.2, 0.4, steps=D*H, **to_double_cuda).reshape(D, H)
# Wh = torch.linspace(-0.4, 0.1, steps=H*H, **to_double_cuda).reshape(H, H)
# b = torch.linspace(-0.7, 0.1, steps=H, **to_double_cuda)
#
# # YOUR_TURN: Impelement rnn_forward
# h, _ = rnn_forward(x, h0, Wx, Wh, b)
# expected_h = torch.tensor([
#   [
#     [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
#     [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
#     [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
#   ],
#   [
#     [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
#     [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
#     [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]], **to_double_cuda)
# print('h error: ', rel_error(expected_h, h))
# ----------------------------------------------------------------------------------------------------
# reset_seed(0)
#
# N, D, T, H = 2, 3, 10, 5
#
# x = torch.randn(N, T, D, **to_double_cuda)
# h0 = torch.randn(N, H, **to_double_cuda)
# Wx = torch.randn(D, H, **to_double_cuda)
# Wh = torch.randn(H, H, **to_double_cuda)
# b = torch.randn(H, **to_double_cuda)
#
# out, cache = rnn_forward(x, h0, Wx, Wh, b)
#
# dout = torch.randn(*out.shape, **to_double_cuda)
#
# # YOUR_TURN: Impelement rnn_backward
# dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
#
# fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
# fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
# fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
# fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
# fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]
#
# dx_num = compute_numeric_gradient(fx, x, dout)
# dh0_num = compute_numeric_gradient(fh0, h0, dout)
# dWx_num = compute_numeric_gradient(fWx, Wx, dout)
# dWh_num = compute_numeric_gradient(fWh, Wh, dout)
# db_num = compute_numeric_gradient(fb, b, dout)
#
# print('dx error: ', rel_error(dx_num, dx))
# print('dh0 error: ', rel_error(dh0_num, dh0))
# print('dWx error: ', rel_error(dWx_num, dWx))
# print('dWh error: ', rel_error(dWh_num, dWh))
# print('db error: ', rel_error(db_num, db))
# ----------------------------------------------------------------------------------------------------
# #
# reset_seed(0)
#
# N, D, T, H = 2, 3, 10, 5
#
# # set requires_grad=True
# x = torch.randn(N, T, D, **to_double_cuda, requires_grad=True)
# h0 = torch.randn(N, H, **to_double_cuda, requires_grad=True)
# Wx = torch.randn(D, H, **to_double_cuda, requires_grad=True)
# Wh = torch.randn(H, H, **to_double_cuda, requires_grad=True)
# b = torch.randn(H, **to_double_cuda, requires_grad=True)
#
# out, cache = rnn_forward(x, h0, Wx, Wh, b)
#
# dout = torch.randn(*out.shape, **to_double_cuda)
#
# # manual backward
# with torch.no_grad():
#   dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
#
# # backward with autograd
# out.backward(dout) # the magic happens here!
# dx_auto, dh0_auto, dWx_auto, dWh_auto, db_auto = \
#   x.grad, h0.grad, Wx.grad, Wh.grad, b.grad
#
# print('dx error: ', rel_error(dx_auto, dx))
# print('dh0 error: ', rel_error(dh0_auto, dh0))
# print('dWx error: ', rel_error(dWx_auto, dWx))
# print('dWh error: ', rel_error(dWh_auto, dWh))
# print('db error: ', rel_error(db_auto, db))
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# model = FeatureExtractor(pooling=True, verbose=True, device='cuda')
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# N, T, V, D = 2, 4, 5, 3
#
# x = torch.tensor([[0, 3, 1, 2], [2, 1, 0, 3]], **to_long_cuda)
# W = torch.linspace(0, 1, steps=V*D, **to_double_cuda).reshape(V, D)
#
# # YOUR_TURN: Impelement WordEmbedding
# model_emb = WordEmbedding(V, D, **to_double_cuda)
# model_emb.W_embed.data.copy_(W)
# out = model_emb(x)
# expected_out = torch.tensor([
#  [[ 0.,          0.07142857,  0.14285714],
#   [ 0.64285714,  0.71428571,  0.78571429],
#   [ 0.21428571,  0.28571429,  0.35714286],
#   [ 0.42857143,  0.5,         0.57142857]],
#  [[ 0.42857143,  0.5,         0.57142857],
#   [ 0.21428571,  0.28571429,  0.35714286],
#   [ 0.,          0.07142857,  0.14285714],
#   [ 0.64285714,  0.71428571,  0.78571429]]], **to_double_cuda)
#
# print('out error: ', rel_error(expected_out, out))
# ----------------------------------------------------------------------------------------------------
# # (Temporal) Affine layer
#
# reset_seed(0)
#
# N, T, D, M = 2, 3, 4, 3
#
# w = torch.linspace(-0.2, 0.4, steps=D*M, **to_double_cuda).reshape(D, M).permute(1, 0)
# b = torch.linspace(-0.4, 0.1, steps=M, **to_double_cuda)
#
# temporal_affine = nn.Linear(D, M).to(**to_double_cuda)
# temporal_affine.weight.data.copy_(w)
# temporal_affine.bias.data.copy_(b)
#
# # For regular affine layer
# x = torch.linspace(-0.1, 0.3, steps=N*D, **to_double_cuda).reshape(N, D)
# out = temporal_affine(x)
# print('affine layer - input shape: {}, output shape: {}'.format(x.shape, out.shape))
# correct_out = torch.tensor([[-0.35584416, -0.10896104,  0.13792208],
#                      [-0.31428571, -0.01753247,  0.27922078]], **to_double_cuda)
#
# print('dx error: ', rel_error(out, correct_out))
#
#
# # For temporal affine layer
# x = torch.linspace(-0.1, 0.3, steps=N*T*D, **to_double_cuda).reshape(N, T, D)
# out = temporal_affine(x)
# print('\ntemporal affine layer - input shape: {}, output shape: {}'.format(x.shape, out.shape))
# correct_out = torch.tensor([[[-0.39920949, -0.16533597,  0.06853755],
#                              [-0.38656126, -0.13750988,  0.11154150],
#                              [-0.37391304, -0.10968379,  0.15454545]],
#                             [[-0.36126482, -0.08185771,  0.19754941],
#                              [-0.34861660, -0.05403162,  0.24055336],
#                              [-0.33596838, -0.02620553,  0.28355731]]], **to_double_cuda)
#
# print('dx error: ', rel_error(out, correct_out))
# ----------------------------------------------------------------------------------------------------
# def check_loss(N, T, V, p):
#     x = 0.001 * torch.randn(N, T, V, **to_double_cuda)
#     y = torch.randint(V, size=(N, T), **to_long_cuda)
#     mask = torch.rand(N, T, **to_double_cuda)
#     y[mask > p] = 0
#     # YOUR_TURN: Impelement temporal_softmax_loss
#     print(temporal_softmax_loss(x, y, NULL_index).item())
#
# check_loss(1000, 1, 10, 1.0)   # Should be about 2.00-2.11
# check_loss(1000, 10, 10, 1.0)  # Should be about 20.6-21.0
# check_loss(5000, 10, 10, 0.1) # Should be about 2.00-2.11
# ----------------------------------------------------------------------------------------------------
# reset_seed(0)
#
# N, D, W, H = 10, 1280, 30, 40
# D_img = 112
# word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
# V = len(word_to_idx)
# T = 13
#
# # YOUR_TURN: Impelement CaptioningRNN
# model = CaptioningRNN(word_to_idx,
#           input_dim=D,
#           wordvec_dim=W,
#           hidden_dim=H,
#           cell_type='rnn',
#           ignore_index=NULL_index,
#           **to_float_cuda) # use float here to be consistent with MobileNet v2
#
#
# for k,v in model.named_parameters():
#   # print(k, v.shape) # uncomment this to see the weight shape
#   v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
# images = torch.linspace(-3., 3., steps=(N * 3 * D_img * D_img),
#                        **to_float_cuda).reshape(N, 3, D_img, D_img)
# captions = (torch.arange(N * T, **to_long_cuda) % V).reshape(N, T)
#
# loss = model(images, captions).item()
# expected_loss = 150.6090393066
#
# print('loss: ', loss)
# print('expected loss: ', expected_loss)
# print('difference: ', rel_error(torch.tensor(loss), torch.tensor(expected_loss)))
# ----------------------------------------------------------------------------------------------------
# def captioning_train(rnn_model, image_data, caption_data, lr_decay=1, **kwargs):
#   """
#   Run optimization to train the model.
#   """
#   # optimizer setup
#   from torch import optim
#   optimizer = optim.Adam(
#     filter(lambda p: p.requires_grad, rnn_model.parameters()),
#     learning_rate) # leave betas and eps by default
#   lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
#                                              lambda epoch: lr_decay ** epoch)
#
#   # sample minibatch data
#   iter_per_epoch = math.ceil(image_data.shape[0] // batch_size)
#   loss_history = []
#   rnn_model.train()
#   for i in range(num_epochs):
#     start_t = time.time()
#     for j in range(iter_per_epoch):
#       images, captions = image_data[j*batch_size:(j+1)*batch_size], \
#                            caption_data[j*batch_size:(j+1)*batch_size]
#
#       loss = rnn_model(images, captions)
#       optimizer.zero_grad()
#       loss.backward()
#       loss_history.append(loss.item())
#       optimizer.step()
#     end_t = time.time()
#     print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
#         i, num_epochs, loss.item(), end_t-start_t))
#
#     lr_scheduler.step()
#
#   # plot the training losses
#   plt.plot(loss_history)
#   plt.xlabel('Iteration')
#   plt.ylabel('Loss')
#   plt.title('Training loss history')
#   plt.show()
#   return rnn_model, loss_history

# ----------------------------------------------------------------------------------------------------

# reset_seed(0)
#
# # data input
# small_num_train = 50
# sample_idx = torch.linspace(0, num_train-1, steps=small_num_train, **to_long_cuda).to('cpu')
# small_image_data = data_dict['train_images'][sample_idx].to('cuda')
# small_caption_data = data_dict['train_captions'][sample_idx].to('cuda')
#
# # optimization arguments
# num_epochs = 80
# batch_size = 50
#
# # create the image captioning model
# model = CaptioningRNN(
#           cell_type='rnn',
#           word_to_idx=data_dict['vocab']['token_to_idx'],
#           input_dim=1280, # hard-coded, do not modify
#           hidden_dim=512,
#           wordvec_dim=256,
#           ignore_index=NULL_index,
#           **to_float_cuda)
#
# for learning_rate in [1e-3]:
#   print('learning rate is: ', learning_rate)
#   rnn_overfit, _ = captioning_train(model, small_image_data, small_caption_data,
#                 num_epochs=num_epochs, batch_size=batch_size,
#                 learning_rate=learning_rate)
# ----------------------------------------------------------------------------------------------------
# reset_seed(0)
#
# # data input
# small_num_train = num_train
# sample_idx = torch.randint(num_train, size=(small_num_train,), **to_long_cuda).to('cpu')
# small_image_data = data_dict['train_images'][sample_idx].to('cuda')
# small_caption_data = data_dict['train_captions'][sample_idx].to('cuda')
#
# # optimization arguments
# num_epochs = 60
# batch_size = 250
#
# # create the image captioning model
# rnn_model = CaptioningRNN(
#           cell_type='rnn',
#           word_to_idx=data_dict['vocab']['token_to_idx'],
#           input_dim=1280, # hard-coded, do not modify
#           hidden_dim=512,
#           wordvec_dim=256,
#           ignore_index=NULL_index,
#           **to_float_cuda)
#
# for learning_rate in [1e-3]:
#   print('learning rate is: ', learning_rate)
#   rnn_model_submit, rnn_loss_submit = captioning_train(rnn_model, small_image_data, small_caption_data,
#                 num_epochs=num_epochs, batch_size=batch_size,
#                 learning_rate=learning_rate)

# ----------------------------------------------------------------------------------------------------
# # Sample a minibatch and show the reshaped 112x112 images,
# # GT captions, and generated captions by your model.
# batch_size = 3
#
# for split in ['train', 'val']:
#   sample_idx = torch.randint(0, num_train if split=='train' else num_val, (batch_size,))
#   sample_images = data_dict[split+'_images'][sample_idx]
#   sample_captions = data_dict[split+'_captions'][sample_idx]
#
#   # decode_captions is loaded from a4_helper.py
#   gt_captions = decode_captions(sample_captions, data_dict['vocab']['idx_to_token'])
#   rnn_model.eval()
#   generated_captions = rnn_model.sample(sample_images)
#   generated_captions = decode_captions(generated_captions, data_dict['vocab']['idx_to_token'])
#
#   for i in range(batch_size):
#     plt.imshow(sample_images[i].permute(1, 2, 0))
#     plt.axis('off')
#     plt.title('%s\nRNN Generated:%s\nGT:%s' % (split, generated_captions[i], gt_captions[i]))
#     plt.show()
# ----------------------------------------------------------------------------------------------------
# N, D, H = 3, 4, 5
# x = torch.linspace(-0.4, 1.2, steps=N*D, **to_double_cuda).reshape(N, D)
# prev_h = torch.linspace(-0.3, 0.7, steps=N*H, **to_double_cuda).reshape(N, H)
# prev_c = torch.linspace(-0.4, 0.9, steps=N*H, **to_double_cuda).reshape(N, H)
# Wx = torch.linspace(-2.1, 1.3, steps=4*D*H, **to_double_cuda).reshape(D, 4 * H)
# Wh = torch.linspace(-0.7, 2.2, steps=4*H*H, **to_double_cuda).reshape(H, 4 * H)
# b = torch.linspace(0.3, 0.7, steps=4*H, **to_double_cuda)
#
# # YOUR_TURN: Impelement lstm_step_forward
# next_h, next_c = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
#
# expected_next_h = torch.tensor([
#     [ 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904],
#     [ 0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ],
#     [ 0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299  ]], **to_double_cuda)
# expected_next_c = torch.tensor([
#     [ 0.32986176,  0.39145139,  0.451556,    0.51014116,  0.56717407],
#     [ 0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345],
#     [ 0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676]], **to_double_cuda)
#
# print('next_h error: ', rel_error(expected_next_h, next_h))
# print('next_c error: ', rel_error(expected_next_c, next_c))
# ----------------------------------------------------------------------------------------------------
N, D, H, T = 2, 5, 4, 3
x = torch.linspace(-0.4, 0.6, steps=N*T*D, **to_double_cuda).reshape(N, T, D)
h0 = torch.linspace(-0.4, 0.8, steps=N*H, **to_double_cuda).reshape(N, H)
Wx = torch.linspace(-0.2, 0.9, steps=4*D*H, **to_double_cuda).reshape(D, 4 * H)
Wh = torch.linspace(-0.3, 0.6, steps=4*H*H, **to_double_cuda).reshape(H, 4 * H)
b = torch.linspace(0.2, 0.7, steps=4*H, **to_double_cuda)

# YOUR_TURN: Impelement lstm_forward
h = lstm_forward(x, h0, Wx, Wh, b)

expected_h = torch.tensor([
 [[ 0.01764008,  0.01823233,  0.01882671,  0.0194232 ],
  [ 0.11287491,  0.12146228,  0.13018446,  0.13902939],
  [ 0.31358768,  0.33338627,  0.35304453,  0.37250975]],
 [[ 0.45767879,  0.4761092,   0.4936887,   0.51041945],
  [ 0.6704845,   0.69350089,  0.71486014,  0.7346449 ],
  [ 0.81733511,  0.83677871,  0.85403753,  0.86935314]]], **to_double_cuda)

print('h error: ', rel_error(expected_h, h))