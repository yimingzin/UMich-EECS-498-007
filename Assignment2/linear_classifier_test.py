import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
import time
import math
import os
from linear_classifier import *

eecs598.reset_seed(0)
data_dict = eecs598.data.preprocess_cifar10(bias_trick=True, cuda=True, dtype=torch.float64)

# print('Train data shape: ', data_dict['X_train'].shape)
# print('Train labels shape: ', data_dict['y_train'].shape)
# print('Validation data shape: ', data_dict['X_val'].shape)
# print('Validation labels shape: ', data_dict['y_val'].shape)
# print('Test data shape: ', data_dict['X_test'].shape)
# print('Test labels shape: ', data_dict['y_test'].shape)
#-----------------------------------------------------------------------------------------------
#
# W = torch.randn(3073, 10, dtype=data_dict['X_val'].dtype, device=data_dict['X_val'].device) * 0.0001
#
# loss, _grad_ = svm_loss_naive(W, data_dict['X_val'], data_dict['y_val'], 0.000005)
# print('loss: %f' % (loss, ))

#-----------------------------------------------------------------------------------------------
# W = 0.0001 * torch.randn(3073, 10, dtype=data_dict['X_val'].dtype, device=data_dict['X_val'].device)
# X_batch = data_dict['X_val'][:128]
# y_batch = data_dict['y_val'][:128]
# reg = 0.000005
#
# # Run and time the naive version
# torch.cuda.synchronize()
# tic = time.time()
# loss_naive, grad_naive = svm_loss_naive(W, X_batch, y_batch, reg)
# torch.cuda.synchronize()
# toc = time.time()
# ms_naive = 1000.0 * (toc - tic)
# print('Naive loss: %e computed in %.2fms' % (loss_naive, ms_naive))
#
# # Run and time the vectorized version
# torch.cuda.synchronize()
# tic = time.time()
# # YOUR_TURN: implement the loss part of 'svm_loss_vectorized' function in "linear_classifier.py"
# loss_vec, _ = svm_loss_vectorized(W, X_batch, y_batch, reg)
# torch.cuda.synchronize()
# toc = time.time()
# ms_vec = 1000.0 * (toc - tic)
# print('Vectorized loss: %e computed in %.2fms' % (loss_vec, ms_vec))
#
# # The losses should match but your vectorized implementation should be much faster.
# print('Difference: %.2e' % (loss_naive - loss_vec))
# print('Speedup: %.2fX' % (ms_naive / ms_vec))
#-----------------------------------------------------------------------------------------------
# torch.cuda.synchronize()
# tic = time.time()
# # 7.5 epoch     num_iters = 1500    每次iteration处理batch_size = 200的数据
# W, loss_hist = train_linear_classifier(svm_loss_vectorized, None,
#                                        data_dict['X_train'],
#                                        data_dict['y_train'],
#                                        learning_rate=3e-11, reg=2.5e4,
#                                        num_iters=1500, verbose=True)
# torch.cuda.synchronize()
# toc = time.time()
# print('That took %fs' % (toc - tic))
#
# plt.plot(loss_hist, 'o')
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()
#
# y_train_pred = predict_linear_classifier(W, data_dict['X_train'])
# train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()
# print('Training accuracy: %.2f%%' % train_acc)
#
# y_val_pred = predict_linear_classifier(W, data_dict['X_val'])
# val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).double().mean().item()
# print('Validation accuracy: %.2f%%' % val_acc)
#-----------------------------------------------------------------------------------------------

### 参考论文
learning_rates, regularization_strengths = svm_get_search_params()
###
num_models = len(learning_rates) * len(regularization_strengths)

####
# It is okay to comment out the following conditions when you are working on svm_get_search_params.
# But, please do not forget to reset back to the original setting once you are done.
if num_models > 25:
  raise Exception("Please do not test/submit more than 25 items at once")
elif num_models < 5:
  raise Exception("Please present at least 5 parameter sets in your final ipynb")
####


i = 0
# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (train_acc, val_acc).
results = {}
best_val = -1.0   # The highest validation accuracy that we have seen so far.
best_svm_model = None # The LinearSVM object that achieved the highest validation rate.
# num_iters = 100 # number of iterations
num_iters = 100

for lr in learning_rates:
  for reg in regularization_strengths:
    i += 1
    print('Training SVM %d / %d with learning_rate=%e and reg=%e'
          % (i, num_models, lr, reg))

    eecs598.reset_seed(0)
    ####
    cand_svm_model, cand_train_acc, cand_val_acc = test_one_param_set(LinearSVM(), data_dict, lr, reg, num_iters)
    ####
    if cand_val_acc > best_val:
      best_val = cand_val_acc
      best_svm_model = cand_svm_model # save the svm
    results[(lr, reg)] = (cand_train_acc, cand_val_acc)


# Print out results.
for lr, reg in sorted(results):
  train_acc, val_acc = results[(lr, reg)]
  print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
         lr, reg, train_acc, val_acc))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# save the best model
path = os.path.join('D:/PythonProject/UMichLearn/Assignment2', 'svm_best_model.pt')
best_svm_model.save(path)

x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')
plt.gcf().set_size_inches(8, 5)
plt.show()

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.gcf().set_size_inches(8, 5)
plt.show()

y_test_pred = best_svm_model.predict(data_dict['X_test'])
test_accuracy = torch.mean((data_dict['y_test'] == y_test_pred).double())
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

w = best_svm_model.W[:-1,:] # strip out the bias
w = w.reshape(3, 32, 32, 10)
w = w.transpose(0, 2).transpose(1, 0)

w_min, w_max = torch.min(w), torch.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
  plt.subplot(2, 5, i + 1)

  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.type(torch.uint8).cpu())
  plt.axis('off')
  plt.title(classes[i])
plt.show()
#-----------------------------------------------------------------------------------------------
#                                           Softmax                                            #
#-----------------------------------------------------------------------------------------------
#
# W = 0.0001 * torch.randn(3073, 10, dtype=data_dict['X_val'].dtype, device=data_dict['X_val'].device)
#
# X_batch = data_dict['X_val'][:128]
# y_batch = data_dict['y_val'][:128]
#
# # YOUR_TURN: Complete the implementation of softmax_loss_naive and implement
# # a (naive) version of the gradient that uses nested loops.
# loss, grad = softmax_loss_naive(W, X_batch, y_batch, reg=10.0)
#
# # As a rough sanity check, our loss should be something close to log(10.0). ( log(C)  = -log(1 / 10), 对于正确类别由于W小接近于e的0次方，其他类别大致相等)
# print('loss: %f' % loss)
# print('sanity check: %f' % (math.log(10.0)))
# f = lambda w: softmax_loss_naive(w, X_batch, y_batch, reg=10.0)[0]
# eecs598.grad.grad_check_sparse(f, W, grad, 10)
#-----------------------------------------------------------------------------------------------

# W = 0.0001 * torch.randn(3073, 10, dtype=data_dict['X_val'].dtype, device=data_dict['X_val'].device)
# reg = 0.05
#
# X_batch = data_dict['X_val'][:128]
# y_batch = data_dict['y_val'][:128]
#
# # Run and time the naive version
# torch.cuda.synchronize()
# tic = time.time()
# loss_naive, grad_naive = softmax_loss_naive(W, X_batch, y_batch, reg)
# torch.cuda.synchronize()
# toc = time.time()
# ms_naive = 1000.0 * (toc - tic)
# print('naive loss: %e computed in %fs' % (loss_naive, ms_naive))
#
# # Run and time the vectorized version
# # YOUR_TURN: Complete the implementation of softmax_loss_vectorized
# torch.cuda.synchronize()
# tic = time.time()
# loss_vec, grad_vec = softmax_loss_vectorized(W, X_batch, y_batch, reg)
# torch.cuda.synchronize()
# toc = time.time()
# ms_vec = 1000.0 * (toc - tic)
# print('vectorized loss: %e computed in %fs' % (loss_vec, ms_vec))
#
# # we use the Frobenius norm to compare the two versions of the gradient.
# loss_diff = (loss_naive - loss_vec).abs().item()
# grad_diff = torch.norm(grad_naive - grad_vec, p='fro')
# print('Loss difference: %.2e' % loss_diff)
# print('Gradient difference: %.2e' % grad_diff)
# print('Speedup: %.2fX' % (ms_naive / ms_vec))
#-----------------------------------------------------------------------------------------------
# device = data_dict['X_train'].device
# dtype = data_dict['X_train'].dtype
# D = data_dict['X_train'].shape[1]
# C = 10
#
# # YOUR_TURN??: train_linear_classifier should be same as what you've implemented in the SVM section
# W_ones = torch.ones(D, C, device=device, dtype=dtype)
# W, loss_hist = train_linear_classifier(softmax_loss_naive, W_ones,
#                                        data_dict['X_train'],
#                                        data_dict['y_train'],
#                                        learning_rate=1e-8, reg=2.5e4,
#                                        num_iters=1, verbose=True)
#
#
# W_ones = torch.ones(D, C, device=device, dtype=dtype)
# W, loss_hist = train_linear_classifier(softmax_loss_vectorized, W_ones,
#                                        data_dict['X_train'],
#                                        data_dict['y_train'],
#                                        learning_rate=1e-8, reg=2.5e4,
#                                        num_iters=1, verbose=True)
#-----------------------------------------------------------------------------------------------
# torch.cuda.synchronize()
# tic = time.time()
#
# # YOUR_TURN: train_linear_classifier should be same as what you've implemented in the SVM section
# W, loss_hist = train_linear_classifier(softmax_loss_vectorized, None,
#                                        data_dict['X_train'],
#                                        data_dict['y_train'],
#                                        learning_rate=1e-10, reg=2.5e4,
#                                        num_iters=1500, verbose=True)
#
# torch.cuda.synchronize()
# toc = time.time()
# print('That took %fs' % (toc - tic))
# plt.plot(loss_hist, 'o')
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()
#
# # evaluate the performance on both the training and validation set
# # YOUR_TURN: predict_linear_classifier should be same as what you've implemented before, in the SVM section
# y_train_pred = predict_linear_classifier(W, data_dict['X_train'])
# train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()
# print('training accuracy: %.2f%%' % train_acc)
# y_val_pred = predict_linear_classifier(W, data_dict['X_val'])
# val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).double().mean().item()
# print('validation accuracy: %.2f%%' % val_acc)

#-----------------------------------------------------------------------------------------------

# # YOUR_TURN: find the best learning_rates and regularization_strengths combination
# #            in 'softmax_get_search_params'
# learning_rates, regularization_strengths = softmax_get_search_params()
# num_models = len(learning_rates) * len(regularization_strengths)
#
# ####
# # It is okay to comment out the following conditions when you are working on svm_get_search_params.
# # But, please do not forget to reset back to the original setting once you are done.
# if num_models > 25:
#   raise Exception("Please do not test/submit more than 25 items at once")
# elif num_models < 5:
#   raise Exception("Please present at least 5 parameter sets in your final ipynb")
# ####
#
#
# i = 0
# # As before, store your cross-validation results in this dictionary.
# # The keys should be tuples of (learning_rate, regularization_strength) and
# # the values should be tuples (train_acc, val_acc)
# results = {}
# best_val = -1.0   # The highest validation accuracy that we have seen so far.
# best_softmax_model = None # The Softmax object that achieved the highest validation rate.
# num_iters = 100 # number of iterations
#
# for lr in learning_rates:
#   for reg in regularization_strengths:
#     i += 1
#     print('Training Softmax %d / %d with learning_rate=%e and reg=%e'
#           % (i, num_models, lr, reg))
#
#     eecs598.reset_seed(0)
#     cand_softmax_model, cand_train_acc, cand_val_acc = test_one_param_set(Softmax(), data_dict, lr, reg, num_iters)
#
#     if cand_val_acc > best_val:
#       best_val = cand_val_acc
#       best_softmax_model = cand_softmax_model # save the classifier
#     results[(lr, reg)] = (cand_train_acc, cand_val_acc)
#
#
# # Print out results.
# for lr, reg in sorted(results):
#   train_acc, val_acc = results[(lr, reg)]
#   print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
#          lr, reg, train_acc, val_acc))
#
# print('best validation accuracy achieved during cross-validation: %f' % best_val)
#
# # save the best model
# path = os.path.join('D:/PythonProject/UMichLearn/Assignment2', 'softmax_best_model.pt')
# best_softmax_model.save(path)
#
# x_scatter = [math.log10(x[0]) for x in results]
# y_scatter = [math.log10(x[1]) for x in results]
#
# # plot training accuracy
# marker_size = 100
# colors = [results[x][0] for x in results]
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 training accuracy')
# plt.gcf().set_size_inches(8, 5)
# plt.show()
#
# # plot validation accuracy
# colors = [results[x][1] for x in results] # default size of markers is 20
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 validation accuracy')
# plt.gcf().set_size_inches(8, 5)
# plt.show()
#
# y_test_pred = best_softmax_model.predict(data_dict['X_test'])
# test_accuracy = torch.mean((data_dict['y_test'] == y_test_pred).double())
# print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))
#
# w = best_softmax_model.W[:-1,:] # strip out the bias
# w = w.reshape(3, 32, 32, 10)
# w = w.transpose(0, 2).transpose(1, 0)
#
# w_min, w_max = torch.min(w), torch.max(w)
#
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for i in range(10):
#   plt.subplot(2, 5, i + 1)
#
#   # Rescale the weights to be between 0 and 255
#   wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
#   plt.imshow(wimg.type(torch.uint8).cpu())
#   plt.axis('off')
#   plt.title(classes[i])
# plt.show()