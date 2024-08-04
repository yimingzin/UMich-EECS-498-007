import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
import time
import math
import os

from eecs598 import reset_seed, Solver
from convolutional_networks import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 16

data_dict = eecs598.data.preprocess_cifar10(cuda=True, dtype=torch.float64, flatten=False)

# ----------------------------------------------------------------------------------------------
# #check Conv forward
# from convolutional_networks import Conv
#
# x_shape = torch.tensor((2, 3, 4, 4))
# w_shape = torch.tensor((3, 3, 4, 4))
# x = torch.linspace(-0.1, 0.5, steps=torch.prod(x_shape), dtype=torch.float64, device='cuda').reshape(*x_shape)
# w = torch.linspace(-0.2, 0.3, steps=torch.prod(w_shape), dtype=torch.float64, device='cuda').reshape(*w_shape)
# b = torch.linspace(-0.1, 0.2, steps=3, dtype=torch.float64, device='cuda')
#
# conv_param = {'stride': 2, 'pad': 1}
# out, _ = Conv.forward(x, w, b, conv_param)
# correct_out = torch.tensor([[[[-0.08759809, -0.10987781],
#                               [-0.18387192, -0.2109216 ]],
#                              [[ 0.21027089,  0.21661097],
#                               [ 0.22847626,  0.23004637]],
#                              [[ 0.50813986,  0.54309974],
#                               [ 0.64082444,  0.67101435]]],
#                             [[[-0.98053589, -1.03143541],
#                               [-1.19128892, -1.24695841]],
#                              [[ 0.69108355,  0.66880383],
#                               [ 0.59480972,  0.56776003]],
#                              [[ 2.36270298,  2.36904306],
#                               [ 2.38090835,  2.38247847]]]],
#                           dtype=torch.float64, device='cuda',
#             )
#
# # Compare your output to ours; difference should be around e-8
# print('Testing Conv.forward')
# print('difference: ', eecs598.grad.rel_error(out, correct_out))
# ----------------------------------------------------------------------------------------------
# 通过卷积操作的前向传播实现图片的灰度转换和边缘检测
# from imageio import imread
# from PIL import Image
# from torchvision.transforms import ToTensor
#
# kitten_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/kitten.jpg'
# puppy_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/puppy.jpg'
#
# kitten = imread(kitten_url)
# puppy = imread(puppy_url)
# # kitten is wide, and puppy is already square
# d = kitten.shape[1] - kitten.shape[0]
# kitten_cropped = kitten[:, d//2:-d//2, :]
#
# img_size = 200   # Make this smaller if it runs too slow
# resized_puppy = ToTensor()(Image.fromarray(puppy).resize((img_size, img_size)))
# resized_kitten = ToTensor()(Image.fromarray(kitten_cropped).resize((img_size, img_size)))
# x = torch.stack([resized_puppy, resized_kitten])
#
# # Set up a convolutional weights holding 2 filters, each 3x3
# w = torch.zeros(2, 3, 3, 3, dtype=x.dtype)
#
# # The first filter converts the image to grayscale.
# # Set up the red, green, and blue channels of the filter.
# w[0, 0, :, :] = torch.tensor([[0, 0, 0], [0, 0.3, 0], [0, 0, 0]])
# w[0, 1, :, :] = torch.tensor([[0, 0, 0], [0, 0.6, 0], [0, 0, 0]])
# w[0, 2, :, :] = torch.tensor([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])
#
# # Second filter detects horizontal edges in the blue channel.
# w[1, 2, :, :] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#
# # Vector of biases. We don't need any bias for the grayscale
# # filter, but for the edge detection filter we want to add 128
# # to each output so that nothing is negative.
# b = torch.tensor([0, 128], dtype=x.dtype)
#
# # Compute the result of convolving each input in x with each filter in w,
# # offsetting by b, and storing the results in out.
# out, _ = Conv.forward(x, w, b, {'stride': 1, 'pad': 1})
#
# def imshow_no_ax(img, normalize=True):
#   """ Tiny helper to show images as uint8 and remove axis labels """
#   if normalize:
#     img_max, img_min = img.max(), img.min()
#     img = 255.0 * (img - img_min) / (img_max - img_min)
#   plt.imshow(img)
#   plt.gca().axis('off')
#
# # Show the original images and the results of the conv operation
# plt.subplot(2, 3, 1)
# imshow_no_ax(puppy, normalize=False)
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# imshow_no_ax(out[0, 0])
# plt.title('Grayscale')
# plt.subplot(2, 3, 3)
# imshow_no_ax(out[0, 1])
# plt.title('Edges')
# plt.subplot(2, 3, 4)
# imshow_no_ax(kitten_cropped, normalize=False)
# plt.subplot(2, 3, 5)
# imshow_no_ax(out[1, 0])
# plt.subplot(2, 3, 6)
# imshow_no_ax(out[1, 1])
# plt.show()
# ----------------------------------------------------------------------------------------------
# # # check Conv backward
# reset_seed(0)
# x = torch.randn(4, 3, 5, 5, dtype=torch.float64, device='cuda')
# w = torch.randn(2, 3, 3, 3, dtype=torch.float64, device='cuda')
# b = torch.randn(2, dtype=torch.float64, device='cuda')
# dout = torch.randn(4, 2, 5, 5, dtype=torch.float64, device='cuda')
# conv_param = {'stride': 1, 'pad': 1}
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: Conv.forward(x, w, b, conv_param)[0], x, dout)
# dw_num = eecs598.grad.compute_numeric_gradient(lambda w: Conv.forward(x, w, b, conv_param)[0], w, dout)
# db_num = eecs598.grad.compute_numeric_gradient(lambda b: Conv.forward(x, w, b, conv_param)[0], b, dout)
#
# out, cache = Conv.forward(x, w, b, conv_param)
# dx, dw, db = Conv.backward(dout, cache)
#
# print('Testing Conv.backward function')
# print('dx error: ', eecs598.grad.rel_error(dx, dx_num))
# print('dw error: ', eecs598.grad.rel_error(dw, dw_num))
# print('db error: ', eecs598.grad.rel_error(db, db_num))
# ----------------------------------------------------------------------------------------------
# # MaxPool forward test
# reset_seed(0)
# x_shape = torch.tensor((2, 3, 4, 4))
# x = torch.linspace(-0.3, 0.4, steps=torch.prod(x_shape), dtype=torch.float64, device='cuda').reshape(*x_shape)
# pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
#
# out, _ = MaxPool.forward(x, pool_param)
#
# correct_out = torch.tensor([[[[-0.26315789, -0.24842105],
#                               [-0.20421053, -0.18947368]],
#                              [[-0.14526316, -0.13052632],
#                               [-0.08631579, -0.07157895]],
#                              [[-0.02736842, -0.01263158],
#                               [ 0.03157895,  0.04631579]]],
#                             [[[ 0.09052632,  0.10526316],
#                               [ 0.14947368,  0.16421053]],
#                              [[ 0.20842105,  0.22315789],
#                               [ 0.26736842,  0.28210526]],
#                              [[ 0.32631579,  0.34105263],
#                               [ 0.38526316,  0.4       ]]]],
#                            dtype=torch.float64, device='cuda')
#
# # Compare your output with ours. Difference should be on the order of e-8.
# print('Testing MaxPool.forward function:')
# print('difference: ', eecs598.grad.rel_error(out, correct_out))
# ----------------------------------------------------------------------------------------------
# Max-pooling: backward test
reset_seed(0)
x = torch.randn(3, 2, 8, 8, dtype=torch.float64, device='cuda')
dout = torch.randn(3, 2, 4, 4, dtype=torch.float64, device='cuda')
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eecs598.grad.compute_numeric_gradient(lambda x: MaxPool.forward(x, pool_param)[0], x, dout)

out, cache = MaxPool.forward(x, pool_param)
dx = MaxPool.backward(dout, cache)

print('Testing MaxPool.backward function:')
print('dx error: ', eecs598.grad.rel_error(dx, dx_num))