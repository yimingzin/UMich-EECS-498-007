import torch

import eecs598
from linear_classifier import svm_loss_naive

data_dict = eecs598.data.preprocess_cifar10(bias_trick=True, cuda=True, dtype=torch.float64)
eecs598.reset_seed(0)
# generate a random SVM weight tensor of small numbers
W = torch.randn(3073, 10, dtype=data_dict['X_val'].dtype, device=data_dict['X_val'].device) * 0.0001

loss, _grad_ = svm_loss_naive(W, data_dict['X_val'], data_dict['y_val'], 0.000005)
print('loss: %f' % (loss, ))