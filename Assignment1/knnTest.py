import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import time

from knn import *

torch.manual_seed(0)
num_train = 5000
num_test = 500
x_train, y_train, x_test, y_test = eecs598.data.cifar10(num_train, num_test)

k_to_accuracies = knn_cross_validate(x_train, y_train, num_folds=5)

for k, accs in sorted(k_to_accuracies.items()):
  print('k = %d got accuracies: %r' % (k, accs))

ks, means, stds = [], [], []
torch.manual_seed(0)
for k, accs in sorted(k_to_accuracies.items()):
  plt.scatter([k] * len(accs), accs, color='g')
  ks.append(k)
  means.append(statistics.mean(accs))
  stds.append(statistics.stdev(accs))
plt.errorbar(ks, means, yerr=stds)
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.title('Cross-validation on k')
plt.show()

from knn import KnnClassifier
from knn import knn_get_best_k

best_k = 1
torch.manual_seed(0)

best_k = knn_get_best_k(k_to_accuracies)
print('Best k is ', best_k)

classifier = KnnClassifier(x_train, y_train)git
classifier.check_accuracy(x_test, y_test, k=best_k)