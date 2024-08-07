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

# # Max-pooling: backward test
# reset_seed(0)
# x = torch.randn(3, 2, 8, 8, dtype=torch.float64, device='cuda')
# dout = torch.randn(3, 2, 4, 4, dtype=torch.float64, device='cuda')
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: MaxPool.forward(x, pool_param)[0], x, dout)
#
# out, cache = MaxPool.forward(x, pool_param)
# dx = MaxPool.backward(dout, cache)
#
# print('Testing MaxPool.backward function:')
# print('dx error: ', eecs598.grad.rel_error(dx, dx_num))

# ----------------------------------------------------------------------------------------------
# # got issue
# # Rel errors should be around e-11 or less
# from convolutional_networks import Conv, FastConv
#
# reset_seed(0)
# x = torch.randn(10, 3, 31, 31, dtype=torch.float64, device='cuda')
# w = torch.randn(25, 3, 3, 3, dtype=torch.float64, device='cuda')
# b = torch.randn(25, dtype=torch.float64, device='cuda')
# dout = torch.randn(10, 25, 16, 16, dtype=torch.float64, device='cuda')
# x_cuda, w_cuda, b_cuda, dout_cuda = x.to('cuda'), w.to('cuda'), b.to('cuda'), dout.to('cuda')
# conv_param = {'stride': 2, 'pad': 1}
#
# t0 = time.time()
# out_naive, cache_naive = Conv.forward(x, w, b, conv_param)
# t1 = time.time()
# out_fast, cache_fast = FastConv.forward(x, w, b, conv_param)
# t2 = time.time()
# out_fast_cuda, cache_fast_cuda = FastConv.forward(x_cuda, w_cuda, b_cuda, conv_param)
# t3 = time.time()
#
# print('Testing FastConv.forward:')
# print('Naive: %fs' % (t1 - t0))
# print('Fast: %fs' % (t2 - t1))
# print('Fast CUDA: %fs' % (t3 - t2))
# print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
# print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
# print('Difference: ', eecs598.grad.rel_error(out_naive, out_fast))
# print('Difference CUDA: ', eecs598.grad.rel_error(out_naive, out_fast_cuda.to(out_naive.device)))
#
#
# t0 = time.time()
# dx_naive, dw_naive, db_naive = Conv.backward(dout, cache_naive)
# t1 = time.time()
# dx_fast, dw_fast, db_fast = FastConv.backward(dout, cache_fast)
# t2 = time.time()
# dx_fast_cuda, dw_fast_cuda, db_fast_cuda = FastConv.backward(dout_cuda, cache_fast_cuda)
# t3 = time.time()
#
# print('\nTesting FastConv.backward:')
# print('Naive: %fs' % (t1 - t0))
# print('Fast: %fs' % (t2 - t1))
# print('Fast CUDA: %fs' % (t3 - t2))
# print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
# print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
# print('dx difference: ', eecs598.grad.rel_error(dx_naive, dx_fast))
# print('dw difference: ', eecs598.grad.rel_error(dw_naive, dw_fast))
# print('db difference: ', eecs598.grad.rel_error(db_naive, db_fast))
# print('dx difference CUDA: ', eecs598.grad.rel_error(dx_naive, dx_fast_cuda.to(dx_naive.device)))
# print('dw difference CUDA: ', eecs598.grad.rel_error(dw_naive, dw_fast_cuda.to(dw_naive.device)))
# print('db difference CUDA: ', eecs598.grad.rel_error(db_naive, db_fast_cuda.to(db_naive.device)))
# ----------------------------------------------------------------------------------------------
# Relative errors should be close to 0.0
# from convolutional_networks import Conv, MaxPool, FastConv, FastMaxPool
#
#
# reset_seed(0)
# x = torch.randn(40, 3, 32, 32, dtype=torch.float64, device='cuda')
# dout = torch.randn(40, 3, 16, 16, dtype=torch.float64, device='cuda')
# x_cuda, dout_cuda = x.to('cuda'), dout.to('cuda')
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
# t0 = time.time()
# out_naive, cache_naive = MaxPool.forward(x, pool_param)
# t1 = time.time()
# out_fast, cache_fast = FastMaxPool.forward(x, pool_param)
# t2 = time.time()
# out_fast_cuda, cache_fast_cuda = FastMaxPool.forward(x_cuda, pool_param)
# t3 = time.time()
#
# print('Testing FastMaxPool.forward:')
# print('Naive: %fs' % (t1 - t0))
# print('Fast: %fs' % (t2 - t1))
# print('Fast CUDA: %fs' % (t3 - t2))
# print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
# print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
# print('Difference: ', eecs598.grad.rel_error(out_naive, out_fast))
# print('Difference CUDA: ', eecs598.grad.rel_error(out_naive, out_fast_cuda.to(out_naive.device)))
#
# t0 = time.time()
# dx_naive = MaxPool.backward(dout, cache_naive)
# t1 = time.time()
# dx_fast = FastMaxPool.backward(dout, cache_fast)
# t2 = time.time()
# dx_fast_cuda = FastMaxPool.backward(dout_cuda, cache_fast_cuda)
# t3 = time.time()
#
# print('\nTesting FastMaxPool.backward:')
# print('Naive: %fs' % (t1 - t0))
# print('Fast: %fs' % (t2 - t1))
# print('Fast CUDA: %fs' % (t3 - t2))
# print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
# print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
# print('dx difference: ', eecs598.grad.rel_error(dx_naive, dx_fast))
# print('dx difference CUDA: ', eecs598.grad.rel_error(dx_naive, dx_fast_cuda.to(dx_naive.device)))
# ----------------------------------------------------------------------------------------------
# from convolutional_networks import Conv_ReLU, Conv_ReLU_Pool
# reset_seed(0)
#
# # Test Conv ReLU
# x = torch.randn(2, 3, 8, 8, dtype=torch.float64, device='cuda')
# w = torch.randn(3, 3, 3, 3, dtype=torch.float64, device='cuda')
# b = torch.randn(3, dtype=torch.float64, device='cuda')
# dout = torch.randn(2, 3, 8, 8, dtype=torch.float64, device='cuda')
# conv_param = {'stride': 1, 'pad': 1}
#
# out, cache = Conv_ReLU.forward(x, w, b, conv_param)
# dx, dw, db = Conv_ReLU.backward(dout, cache)
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: Conv_ReLU.forward(x, w, b, conv_param)[0], x, dout)
# dw_num = eecs598.grad.compute_numeric_gradient(lambda w: Conv_ReLU.forward(x, w, b, conv_param)[0], w, dout)
# db_num = eecs598.grad.compute_numeric_gradient(lambda b: Conv_ReLU.forward(x, w, b, conv_param)[0], b, dout)
#
# # Relative errors should be around e-8 or less
# print('Testing Conv_ReLU:')
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
# print('dw error: ', eecs598.grad.rel_error(dw_num, dw))
# print('db error: ', eecs598.grad.rel_error(db_num, db))
#
# # Test Conv ReLU Pool
# x = torch.randn(2, 3, 16, 16, dtype=torch.float64, device='cuda')
# w = torch.randn(3, 3, 3, 3, dtype=torch.float64, device='cuda')
# b = torch.randn(3, dtype=torch.float64, device='cuda')
# dout = torch.randn(2, 3, 8, 8, dtype=torch.float64, device='cuda')
# conv_param = {'stride': 1, 'pad': 1}
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
# out, cache = Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)
# dx, dw, db = Conv_ReLU_Pool.backward(dout, cache)
#
# dx_num = eecs598.grad.compute_numeric_gradient(lambda x: Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)[0], x, dout)
# dw_num = eecs598.grad.compute_numeric_gradient(lambda w: Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)[0], w, dout)
# db_num = eecs598.grad.compute_numeric_gradient(lambda b: Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)[0], b, dout)
#
# # Relative errors should be around e-8 or less
# print()
# print('Testing Conv_ReLU_Pool')
# print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
# print('dw error: ', eecs598.grad.rel_error(dw_num, dw))
# print('db error: ', eecs598.grad.rel_error(db_num, db))
# ----------------------------------------------------------------------------------------------
# from convolutional_networks import ThreeLayerConvNet
#
# reset_seed(0)
# model = ThreeLayerConvNet(dtype=torch.float64, device='cuda')
#
# N = 50
# X = torch.randn(N, 3, 32, 32, dtype=torch.float64, device='cuda')
# y = torch.randint(10, size=(N,), dtype=torch.int64, device='cuda')
#
# loss, grads = model.loss(X, y)
# print('Initial loss (no regularization): ', loss.item())
#
# model.reg = 0.5
# loss, grads = model.loss(X, y)
# print('Initial loss (with regularization): ', loss.item())
#
# from convolutional_networks import ThreeLayerConvNet
#
# num_inputs = 2
# input_dims = (3, 16, 16)
# reg = 0.0
# num_classes = 10
# reset_seed(0)
# X = torch.randn(num_inputs, *input_dims, dtype=torch.float64, device='cuda')
# y = torch.randint(num_classes, size=(num_inputs,), dtype=torch.int64, device='cuda')
#
# model = ThreeLayerConvNet(num_filters=3, filter_size=3,
#                           input_dims=input_dims, hidden_dim=7,
#                           weight_scale=5e-2, dtype=torch.float64, device='cuda')
# loss, grads = model.loss(X, y)
#
# for param_name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     param_grad_num = eecs598.grad.compute_numeric_gradient(f, model.params[param_name])
#     print('%s max relative error: %e' % (param_name, eecs598.grad.rel_error(param_grad_num, grads[param_name])))
# ----------------------------------------------------------------------------------------------
# # train one small data got overfit
# from convolutional_networks import ThreeLayerConvNet
# from fully_connected_networks import adam
#
# reset_seed(0)
#
# num_train = 100
# small_data = {
#   'X_train': data_dict['X_train'][:num_train],
#   'y_train': data_dict['y_train'][:num_train],
#   'X_val': data_dict['X_val'],
#   'y_val': data_dict['y_val'],
# }
#
# model = ThreeLayerConvNet(weight_scale=1e-3, dtype=torch.float32, device='cuda')
#
# solver = Solver(model, small_data,
#                 num_epochs=30, batch_size=50,
#                 update_rule=adam,
#                 optim_config={
#                   'learning_rate': 2e-3,
#                 },
#                 verbose=True, print_every=1,
#                 device='cuda')
# solver.train()
#
# plt.title('Training losses')
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.gcf().set_size_inches(9, 4)
# plt.show()
#
# plt.title('Train and Val accuracies')
# plt.plot(solver.train_acc_history, '-o')
# plt.plot(solver.val_acc_history, '-o')
# plt.legend(['train', 'val'], loc='upper left')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.gcf().set_size_inches(9, 4)
# plt.show()
# ----------------------------------------------------------------------------------------------
# from convolutional_networks import ThreeLayerConvNet
# from fully_connected_networks import adam
#
# reset_seed(0)
#
# model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, dtype=torch.float, device='cuda')
#
# solver = Solver(model, data_dict,
#                 num_epochs=1, batch_size=64,
#                 update_rule=adam,
#                 optim_config={
#                   'learning_rate': 2e-3,
#                 },
#                 verbose=True, print_every=50, device='cuda')
# solver.train()
#
# from torchvision.utils import make_grid
# nrow = math.ceil(math.sqrt(model.params['W1'].shape[0]))
# grid = make_grid(model.params['W1'], nrow=nrow, padding=1, normalize=True, scale_each=True)
# plt.imshow(grid.to(device='cpu').permute(1, 2, 0))
# plt.axis('off')
# plt.gcf().set_size_inches(5, 5)
# plt.show()
# ----------------------------------------------------------------------------------------------
# from convolutional_networks import DeepConvNet
# from fully_connected_networks import adam
#
# reset_seed(0)
# num_inputs = 2
# input_dims = (3, 8, 8)
# num_classes = 10
# N = 50
#
# X = torch.randn(N, *input_dims, dtype=torch.float64, device='cuda')
# y = torch.randint(10, size=(N,), dtype=torch.int64, device='cuda')
#
# for reg in [0, 3.14]:
#   print('Running check with reg = ', reg)
#   model = DeepConvNet(input_dims=input_dims, num_classes=num_classes,
#                       num_filters=[8, 8, 8],
#                       max_pools=[0, 2],
#                       reg=reg,
#                       weight_scale=5e-2, dtype=torch.float64, device='cuda')
#
#   loss, grads = model.loss(X, y)
#   # The relative errors should be up to the order of e-6
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eecs598.grad.compute_numeric_gradient(f, model.params[name])
#     print('%s max relative error: %e' % (name, eecs598.grad.rel_error(grad_num, grads[name])))
#   if reg == 0: print()
# ----------------------------------------------------------------------------------------------
# # TODO: Use a DeepConvNet to overfit 50 training examples by
# # tweaking just the learning rate and initialization scale.
# from convolutional_networks import DeepConvNet, find_overfit_parameters
# from fully_connected_networks import adam
#
# reset_seed(0)
# num_train = 50
# small_data = {
#   'X_train': data_dict['X_train'][:num_train],
#   'y_train': data_dict['y_train'][:num_train],
#   'X_val': data_dict['X_val'],
#   'y_val': data_dict['y_val'],
# }
# input_dims = small_data['X_train'].shape[1:]
#
#
# # Update the parameters in find_overfit_parameters in convolutional_networks.py
# weight_scale, learning_rate = find_overfit_parameters()
#
#
# model = DeepConvNet(input_dims=input_dims, num_classes=10,
#                     num_filters=[8, 16, 32, 64],
#                     max_pools=[0, 1, 2, 3],
#                     reg=1e-5, weight_scale=weight_scale, dtype=torch.float32, device='cuda')
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=30, batch_size=10,
#                 update_rule=adam,
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 },
#                 device='cuda',
#          )
# # Turn off keep_best_params to allow final weights to be saved, instead of best weights on validation set.
# solver.train(return_best_params=False)
#
# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()
#
# val_acc = solver.check_accuracy(
#                         solver.X_train, solver.y_train, num_samples=solver.num_train_samples
#                     )
# print(val_acc)
#
# path = os.path.join('D:/PythonProject/UMichLearn/Assignment3', 'overfit_deepconvnet.pth')
# solver.model.save(path)
# # Create a new instance
# model = DeepConvNet(input_dims=input_dims, num_classes=10,
#                     num_filters=[8, 16, 32, 64],
#                     max_pools=[0, 1, 2, 3],
#                     reg=1e-5, weight_scale=weight_scale, dtype=torch.float32, device='cuda')
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=30, batch_size=10,
#                 update_rule=adam,
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 },
#                 device='cuda',
#          )
#
# # Load model
# solver.model.load(path, dtype=torch.float32, device='cuda')
#
# # Evaluate on validation set
# accuracy = solver.check_accuracy(small_data['X_train'], small_data['y_train'])
# print(f"Saved model's accuracy on training is {accuracy}")
# ----------------------------------------------------------------------------------------------
# # Check the training-time forward pass by checking means and variances
# # of features both before and after batch normalization
# from convolutional_networks import BatchNorm
#
# def print_mean_std(x,dim=0):
#   means = ['%.3f' % xx for xx in x.mean(dim=dim).tolist()]
#   stds = ['%.3f' % xx for xx in x.std(dim=dim).tolist()]
#   print('  means: ', means)
#   print('  stds:  ', stds)
#   print()
#
# # Simulate the forward pass for a two-layer network
# reset_seed(0)
# N, D1, D2, D3 = 200, 50, 60, 3
# X = torch.randn(N, D1, dtype=torch.float64, device='cuda')
# W1 = torch.randn(D1, D2, dtype=torch.float64, device='cuda')
# W2 = torch.randn(D2, D3, dtype=torch.float64, device='cuda')
# a = X.matmul(W1).clamp(min=0.).matmul(W2)
#
# print('Before batch normalization:')
# print_mean_std(a,dim=0)
#
# # Run with gamma=1, beta=0. Means should be close to zero and stds close to one
# gamma = torch.ones(D3, dtype=torch.float64, device='cuda')
# beta = torch.zeros(D3, dtype=torch.float64, device='cuda')
# print('After batch normalization (gamma=1, beta=0)')
# a_norm, _ = BatchNorm.forward(a, gamma, beta, {'mode': 'train'})
# print_mean_std(a_norm,dim=0)
#
# # Run again with nontrivial gamma and beta. Now means should be close to beta
# # and std should be close to gamma.
# gamma = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device='cuda')
# beta = torch.tensor([11.0, 12.0, 13.0], dtype=torch.float64, device='cuda')
# print('After batch normalization (gamma=', gamma.tolist(), ', beta=', beta.tolist(), ')')
# a_norm, _ = BatchNorm.forward(a, gamma, beta, {'mode': 'train'})
# print_mean_std(a_norm,dim=0)
#
#
# from convolutional_networks import BatchNorm
#
# reset_seed(0)
# N, D1, D2, D3 = 200, 50, 60, 3
# W1 = torch.randn(D1, D2, dtype=torch.float64, device='cuda')
# W2 = torch.randn(D2, D3, dtype=torch.float64, device='cuda')
#
# bn_param = {'mode': 'train'}
# gamma = torch.ones(D3, dtype=torch.float64, device='cuda')
# beta = torch.zeros(D3, dtype=torch.float64, device='cuda')
#
# for t in range(500):
#   X = torch.randn(N, D1, dtype=torch.float64, device='cuda')
#   a = X.matmul(W1).clamp(min=0.).matmul(W2)
#   BatchNorm.forward(a, gamma, beta, bn_param)
#
# bn_param['mode'] = 'test'
# X = torch.randn(N, D1, dtype=torch.float64, device='cuda')
# a = X.matmul(W1).clamp(min=0.).matmul(W2)
# a_norm, _ = BatchNorm.forward(a, gamma, beta, bn_param)
#
# # Means should be close to zero and stds close to one, but will be
# # noisier than training-time forward passes.
# print('After batch normalization (test-time):')
# print_mean_std(a_norm,dim=0)
# ----------------------------------------------------------------------------------------------
from convolutional_networks import BatchNorm

# Gradient check batchnorm backward pass
reset_seed(0)
N, D = 4, 5
x = 5 * torch.randn(N, D, dtype=torch.float64, device='cuda') + 12
gamma = torch.randn(D, dtype=torch.float64, device='cuda')
beta = torch.randn(D, dtype=torch.float64, device='cuda')
dout = torch.randn(N, D, dtype=torch.float64, device='cuda')

bn_param = {'mode': 'train'}
fx = lambda x: BatchNorm.forward(x, gamma, beta, bn_param)[0]
fg = lambda a: BatchNorm.forward(x, a, beta, bn_param)[0]
fb = lambda b: BatchNorm.forward(x, gamma, b, bn_param)[0]

dx_num = eecs598.grad.compute_numeric_gradient(fx, x, dout)
da_num = eecs598.grad.compute_numeric_gradient(fg, gamma.clone(), dout)
db_num = eecs598.grad.compute_numeric_gradient(fb, beta.clone(), dout)

_, cache = BatchNorm.forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
# You should expect to see relative errors between 1e-12 and 1e-9
print('dx error: ', eecs598.grad.rel_error(dx_num, dx))
print('dgamma error: ', eecs598.grad.rel_error(da_num, dgamma))
print('dbeta error: ', eecs598.grad.rel_error(db_num, dbeta))