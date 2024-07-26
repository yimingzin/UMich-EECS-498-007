import os

import eecs598
import torch
import matplotlib.pyplot as plt
import statistics
import time
from eecs598.a2_helpers import get_toy_data, plot_stats, show_net_weights, plot_acc_curves
from two_layer_net import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 16
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ----------------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------
# #
# eecs598.reset_seed(0)
#
# reg = 0.05
# toy_X, toy_y, params = get_toy_data(dtype=torch.float64)
#
# # YOUR_TURN: Implement the gradient computation part of nn_forward_backward
# #            When you implement the gradient computation part, you may need to
# #            implement the `hidden` output in nn_forward_pass, as well.
# loss, grads = nn_forward_backward(params, toy_X, toy_y, reg=reg)
#
# for param_name, grad in grads.items():
#   param = params[param_name]
#   f = lambda w: nn_forward_backward(params, toy_X, toy_y, reg=reg)[0]
#   grad_numeric = eecs598.grad.compute_numeric_gradient(f, param)
#   error = eecs598.grad.rel_error(grad, grad_numeric)
#   print('%s max relative error: %e' % (param_name, error))

# ----------------------------------------------------------------------------------------------------------
# eecs598.reset_seed(0)
# toy_X, toy_y, params = get_toy_data()
#
# # YOUR_TURN: Implement the nn_train function.
# #            You may need to check nn_predict function (the "pred_func") as well.
# stats = nn_train(params, nn_forward_backward, nn_predict, toy_X, toy_y, toy_X, toy_y,
#                  learning_rate=1e-1, reg=1e-6,
#                  num_iters=200, verbose=False)
#
# print('Final training loss: ', stats['loss_history'][-1])
#
# # plot the loss history
# plt.plot(stats['loss_history'], 'o')
# plt.xlabel('Iteration')
# plt.ylabel('training loss')
# plt.title('Training Loss history')
# plt.show()
# # Plot the loss function and train / validation accuracies
# plt.plot(stats['train_acc_history'], 'o', label='train')
# plt.plot(stats['val_acc_history'], 'o', label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.legend()
# plt.show()

# ----------------------------------------------------------------------------------------------------------
# # Invoke the above function to get our data.
# eecs598.reset_seed(0)
# data_dict = eecs598.data.preprocess_cifar10(dtype=torch.float64)
#
# input_size = 3 * 32 * 32
# hidden_size = 36
# num_classes = 10
#
# # fix random seed before we generate a set of parameters
# eecs598.reset_seed(0)
# net = TwoLayerNet(input_size, hidden_size, num_classes, dtype=data_dict['X_train'].dtype, device=data_dict['X_train'].device)
#
# # Train the network
# stats = net.train(data_dict['X_train'], data_dict['y_train'],
#                   data_dict['X_val'], data_dict['y_val'],
#                   num_iters=500, batch_size=1000,
#                   learning_rate=1e-2, learning_rate_decay=0.95,
#                   reg=0.25, verbose=True)
#
# # Predict on the validation set
# y_val_pred = net.predict(data_dict['X_val'])
# val_acc = 100.0 * (y_val_pred == data_dict['y_val']).double().mean().item()
# print('Validation accuracy: %.2f%%' % val_acc)
#
# plot_stats(stats)
# show_net_weights(net)
# ----------------------------------------------------------------------------------------------------------
# 测试不同的隐藏层size
# eecs598.reset_seed(0)
# data_dict = eecs598.data.preprocess_cifar10(dtype=torch.float64)
# hidden_sizes = [2, 8, 32, 128]
# lr = 0.1
# reg = 0.001
#
# stat_dict = {}
# for hs in hidden_sizes:
#   print('train with hidden size: {}'.format(hs))
#   # fix random seed before we generate a set of parameters
#   eecs598.reset_seed(0)
#   net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device, dtype=data_dict['X_train'].dtype)
#   stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
#             num_iters=3000, batch_size=1000,
#             learning_rate=lr, learning_rate_decay=0.95,
#             reg=reg, verbose=False)
#   stat_dict[hs] = stats
#   plot_acc_curves(stat_dict)
# ----------------------------------------------------------------------------------------------------------
# eecs598.reset_seed(0)
# data_dict = eecs598.data.preprocess_cifar10(dtype=torch.float64)
# hs = 128
# lr = 1.0
# regs = [0, 1e-5, 1e-3, 1e-1]
#
# stat_dict = {}
# for reg in regs:
#   print('train with regularization: {}'.format(reg))
#   # fix random seed before we generate a set of parameters
#   eecs598.reset_seed(0)
#   net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device, dtype=data_dict['X_train'].dtype)
#   stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
#             num_iters=3000, batch_size=1000,
#             learning_rate=lr, learning_rate_decay=0.95,
#             reg=reg, verbose=False)
#   stat_dict[reg] = stats
#
# plot_acc_curves(stat_dict)
# ----------------------------------------------------------------------------------------------------------
# eecs598.reset_seed(0)
# data_dict = eecs598.data.preprocess_cifar10(dtype=torch.float64)
# hs = 128
# lrs = [1e-4, 1e-2, 1e0, 1e2]
# reg = 1e-4
#
# stat_dict = {}
# for lr in lrs:
#   print('train with learning rate: {}'.format(lr))
#   # fix random seed before we generate a set of parameters
#   eecs598.reset_seed(0)
#   net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device, dtype=data_dict['X_train'].dtype)
#   stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
#             num_iters=3000, batch_size=1000,
#             learning_rate=lr, learning_rate_decay=0.95,
#             reg=reg, verbose=False)
#   stat_dict[lr] = stats
#
# plot_acc_curves(stat_dict)
# ----------------------------------------------------------------------------------------------------------
# running this model on float64 may needs more time, so set it as float32
eecs598.reset_seed(0)
data_dict = eecs598.data.preprocess_cifar10(dtype=torch.float32)

# store the best model into this
eecs598.reset_seed(0)
best_net, best_stat, best_val_acc = find_best_net(data_dict, nn_get_search_params)
print(best_val_acc)

plot_stats(best_stat)

# save the best model
path = os.path.join('D:/PythonProject/UMichLearn/Assignment2', 'nn_best_model.pt')
best_net.save(path)

# Check the validation-set accuracy of your best model
y_val_preds = best_net.predict(data_dict['X_val'])
val_acc = 100 * (y_val_preds == data_dict['y_val']).double().mean().item()
print('Best val-set accuracy: %.2f%%' % val_acc)

# visualize the weights of the best network
show_net_weights(best_net)

y_test_preds = best_net.predict(data_dict['X_test'])
test_acc = 100 * (y_test_preds == data_dict['y_test']).double().mean().item()
print('Test accuracy: %.2f%%' % test_acc)