import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
import time
import math
from a3_helper import svm_loss, softmax_loss
from eecs598 import reset_seed, Solver
from fully_connected_networks import *


plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# ----------------------------------------------------------------------------------------------
# # Test the Linear.forward function
# num_inputs = 2
# input_shape = torch.tensor((4, 5, 6))
# output_dim = 3
#
# input_size = num_inputs * torch.prod(input_shape)
# weight_size = output_dim * torch.prod(input_shape)
#
# x = torch.linspace(-0.1, 0.5, steps=input_size, dtype=torch.float64, device='cuda')
# w = torch.linspace(-0.2, 0.3, steps=weight_size, dtype=torch.float64, device='cuda')
# b = torch.linspace(-0.3, 0.1, steps=output_dim, dtype=torch.float64, device='cuda')
# x = x.reshape(num_inputs, *input_shape)
# w = w.reshape(torch.prod(input_shape), output_dim)
#
# out, _ = Linear.forward(x, w, b)
# correct_out = torch.tensor([[1.49834984, 1.70660150, 1.91485316],
#                             [3.25553226, 3.51413301, 3.77273372]]
#                             ).double().cuda()
#
# print('Testing Linear.forward function:')
# print('difference: ', eecs598.grad.rel_error(out, correct_out))

# ----------------------------------------------------------------------------------------------
# # Test the Linear.backward function
# reset_seed(0)
# x = torch.randn(10, 2, 3, dtype=torch.float64, device='cuda')
# w = torch.randn(6, 5, dtype=torch.float64, device='cuda')
# b = torch.randn(5, dtype=torch.float64, device='cuda')
# dout = torch.randn(10, 5, dtype=torch.float64, device='cuda')
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: Linear.forward(x, w, b)[0], x, dout)
# dw_num = eecs598.grad.compute_numeric_gradient(lambda w: Linear.forward(x, w, b)[0], w, dout)
# db_num = eecs598.grad.compute_numeric_gradient(lambda b: Linear.forward(x, w, b)[0], b, dout)
#
# _, cache = Linear.forward(x, w, b)
# dx, dw, db = Linear.backward(dout, cache)
#
# # The error should be around e-10 or less
# print('Testing Linear.backward function:')
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
# print('dw error: ', eecs598.grad.rel_error(dw_num, dw))
# print('db error: ', eecs598.grad.rel_error(db_num, db))

# ----------------------------------------------------------------------------------------------
# reset_seed(0)
# x = torch.linspace(-0.5, 0.5, steps=12, dtype=torch.float64, device='cuda')
# x = x.reshape(3, 4)
#
# out, _ = ReLU.forward(x)
# correct_out = torch.tensor([[ 0.,          0.,          0.,          0.,        ],
#                             [ 0.,          0.,          0.04545455,  0.13636364,],
#                             [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]],
#                             dtype=torch.float64,
#                             device='cuda')
#
# # Compare your output with ours. The error should be on the order of e-8
# print('Testing ReLU.forward function:')
# print('difference: ', eecs598.grad.rel_error(out, correct_out))
# ----------------------------------------------------------------------------------------------
# reset_seed(0)
# x = torch.randn(10, 10, dtype=torch.float64, device='cuda')
# dout = torch.randn(*x.shape, dtype=torch.float64, device='cuda')
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: ReLU.forward(x)[0], x, dout)
#
# _, cache = ReLU.forward(x)
# dx = ReLU.backward(dout, cache)
#
# # The error should be on the order of e-12
# print('Testing ReLU.backward function:')
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
# ----------------------------------------------------------------------------------------------

# reset_seed(0)
# x = torch.randn(2, 3, 4, dtype=torch.float64, device='cuda')
# w = torch.randn(12, 10, dtype=torch.float64, device='cuda')
# b = torch.randn(10, dtype=torch.float64, device='cuda')
# dout = torch.randn(2, 10, dtype=torch.float64, device='cuda')
#
# out, cache = Linear_ReLU.forward(x, w, b)
# dx, dw, db = Linear_ReLU.backward(dout, cache)
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: Linear_ReLU.forward(x, w, b)[0], x, dout)
# dw_num = eecs598.grad.compute_numeric_gradient(lambda w: Linear_ReLU.forward(x, w, b)[0], w, dout)
# db_num = eecs598.grad.compute_numeric_gradient(lambda b: Linear_ReLU.forward(x, w, b)[0], b, dout)
#
# # Relative error should be around e-8 or less
# print('Testing Linear_ReLU.forward and Linear_ReLU.backward:')
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
# print('dw error: ', eecs598.grad.rel_error(dw_num, dw))
# print('db error: ', eecs598.grad.rel_error(db_num, db))

# ----------------------------------------------------------------------------------------------
# reset_seed(0)
# num_classes, num_inputs = 10, 50
# x = 0.001 * torch.randn(num_inputs, num_classes, dtype=torch.float64, device='cuda')
# y = torch.randint(num_classes, size=(num_inputs,), dtype=torch.int64, device='cuda')
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: svm_loss(x, y)[0], x)
# loss, dx = svm_loss(x, y)
#
# # Test svm_loss function. Loss should be around 9 and dx error should be around the order of e-9
# print('Testing svm_loss:')
# print('loss: ', loss.item())
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: softmax_loss(x, y)[0], x)
# loss, dx = softmax_loss(x, y)
#
# # Test softmax_loss function. Loss should be close to 2.3 and dx error should be around e-8
# print('\nTesting softmax_loss:')
# print('loss: ', loss.item())
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
# ----------------------------------------------------------------------------------------------
# reset_seed(0)
# N, D, H, C = 3, 5, 50, 7
# X = torch.randn(N, D, dtype=torch.float64, device='cuda')
# y = torch.randint(C, size=(N,), dtype=torch.int64, device='cuda')
#
# std = 1e-3
# model = TwoLayerNet(
#           input_dim=D,
#           hidden_dim=H,
#           num_classes=C,
#           weight_scale=std,
#           dtype=torch.float64,
#           device='cuda'
#         )
#
# print('Testing initialization ... ')
# W1_std = torch.abs(model.params['W1'].std() - std)
# b1 = model.params['b1']
# W2_std = torch.abs(model.params['W2'].std() - std)
# b2 = model.params['b2']
# assert W1_std < std / 10, 'First layer weights do not seem right'
# assert torch.all(b1 == 0), 'First layer biases do not seem right'
# assert W2_std < std / 10, 'Second layer weights do not seem right'
# assert torch.all(b2 == 0), 'Second layer biases do not seem right'
#
# print('Testing test-time forward pass ... ')
# model.params['W1'] = torch.linspace(-0.7, 0.3, steps=D * H, dtype=torch.float64, device='cuda').reshape(D, H)
# model.params['b1'] = torch.linspace(-0.1, 0.9, steps=H, dtype=torch.float64, device='cuda')
# model.params['W2'] = torch.linspace(-0.3, 0.4, steps=H * C, dtype=torch.float64, device='cuda').reshape(H, C)
# model.params['b2'] = torch.linspace(-0.9, 0.1, steps=C, dtype=torch.float64, device='cuda')
# X = torch.linspace(-5.5, 4.5, steps=N * D, dtype=torch.float64, device='cuda').reshape(D, N).t()
# scores = model.loss(X)
# correct_scores = torch.tensor(
#   [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
#    [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
#    [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]],
#     dtype=torch.float64, device='cuda')
# scores_diff = torch.abs(scores - correct_scores).sum()
# assert scores_diff < 1e-6, 'Problem with test-time forward pass'
#
# print('Testing training loss (no regularization)')
# y = torch.tensor([0, 5, 1])
# loss, grads = model.loss(X, y)
# correct_loss = 3.4702243556
# assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
#
# model.reg = 1.0
# loss, grads = model.loss(X, y)
# correct_loss = 49.719461034881775
# assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
#
# # Errors should be around e-6 or lessb
# for reg in [0.0, 0.7]:
#   print('Running numeric gradient check with reg = ', reg)
#   model.reg = reg
#   loss, grads = model.loss(X, y)
#
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eecs598.grad.compute_numeric_gradient(f, model.params[name])
#     print('%s relative error: %.2e' % (name, eecs598.grad.rel_error(grad_num, grads[name])))