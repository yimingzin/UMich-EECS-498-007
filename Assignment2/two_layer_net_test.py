import os
import eecs598
import torch
import matplotlib.pyplot as plt
import statistics
import random
import time

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 16
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ----------------------------------------------------------------------------------------------------
# import eecs598
# from eecs598.a2_helpers import get_toy_data
# from two_layer_net import nn_forward_pass
#
# eecs598.reset_seed(0)
# toy_X, toy_y, params = get_toy_data()
#
# # YOUR_TURN: Implement the score computation part of nn_forward_pass
# scores, _ = nn_forward_pass(params, toy_X)
# print('Your scores:')
# print(scores)
# print(scores.dtype)
# print()
# print('correct scores:')
# correct_scores = torch.tensor([
#         [ 9.7003e-08, -1.1143e-07, -3.9961e-08],
#         [-7.4297e-08,  1.1502e-07,  1.5685e-07],
#         [-2.5860e-07,  2.2765e-07,  3.2453e-07],
#         [-4.7257e-07,  9.0935e-07,  4.0368e-07],
#         [-1.8395e-07,  7.9303e-08,  6.0360e-07]], dtype=torch.float32, device=scores.device)
# print(correct_scores)
# print()
#
# # The difference should be very small. We get < 1e-10
# scores_diff = (scores - correct_scores).abs().sum().item()
# print('Difference between your scores and correct scores: %.2e' % scores_diff)
# ----------------------------------------------------------------------------------------------------

# import eecs598
# from eecs598.a2_helpers import get_toy_data
# from two_layer_net import nn_forward_backward
#
# eecs598.reset_seed(0)
# toy_X, toy_y, params = get_toy_data()
#
# # YOUR_TURN: Implement the loss computation part of nn_forward_backward
# loss, _ = nn_forward_backward(params, toy_X, toy_y, reg=0.05)
# print('Your loss: ', loss.item())
# correct_loss = 1.0986121892929077
# print('Correct loss: ', correct_loss)
# diff = (correct_loss - loss).item()
#
# # should be very small, we get < 1e-4
# print('Difference: %.4e' % diff)

# ----------------------------------------------------------------------------------------------------
import eecs598
from eecs598.a2_helpers import get_toy_data
from two_layer_net import nn_forward_backward

eecs598.reset_seed(0)

reg = 0.05
toy_X, toy_y, params = get_toy_data(dtype=torch.float64)

# YOUR_TURN: Implement the gradient computation part of nn_forward_backward
#            When you implement the gradient computation part, you may need to
#            implement the `hidden` output in nn_forward_pass, as well.
loss, grads = nn_forward_backward(params, toy_X, toy_y, reg=reg)

for param_name, grad in grads.items():
  param = params[param_name]
  f = lambda w: nn_forward_backward(params, toy_X, toy_y, reg=reg)[0]
  grad_numeric = eecs598.grad.compute_numeric_gradient(f, param)
  error = eecs598.grad.rel_error(grad, grad_numeric)
  print('%s max relative error: %e' % (param_name, error))