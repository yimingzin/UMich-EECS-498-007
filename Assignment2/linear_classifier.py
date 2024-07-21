import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional

"""
    支持向量机损失函数朴素实现
"""
def svm_loss_naive(
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        reg: float
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]
#   外层循环遍历num_train
    for i in range(num_train):
        #计算当前num_train在所有种类的得分
        scores = torch.mv(W.t(), X[i])
        #正确标签得分
        correct_scores = scores[y[i]]
        for j in range(num_class):
            #计算损失函数只计算错误标签
            if j == y[i]:
                continue
            # SVM Loss - 计算间隔，当前num_train的其他类别得分 - 正确类别得分, 看间隔是否大于1
            margin = scores[j] - correct_scores + 1
            # 更新损失和梯度, 损失即直接加间隔, 梯度求导后其他类别相加，正确类别相减
            if margin > 0 :
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    #正则化
    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW

def softmax_loss_naive(
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        reg: float
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        scores = torch.mv(W.t(), X[i])
        #上面同svm实现，softmax计算概率，需要保证数字稳定性即减去每个num_train中最大的得分
        scores_stable = scores - scores.max()
        correct_scores = scores_stable[y[i]]
        sum_exp = torch.sum(torch.exp(scores_stable))
        #softmax的损失函数 = -log(e^(正确类别得分) / e^(所有类别得分)求和 )
        loss += torch.log(sum_exp) - correct_scores
        #梯度计算，对于正确类别需要-1，其他类别直接计算即可 - 核心是e^(类别得分) / e^(所有类别得分)求和，这里把正确类别和错误类别分开讨论
        for j in range(num_class):
            if j == y[i]:
                dW[:, j] += (torch.exp(scores_stable[j])/ sum_exp - 1) * X[i]
            else:
                dW[:, j] += torch.exp(scores_stable[j]) / sum_exp * X[i]

    #正则化
    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW

def svm_loss_vectorized(
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        reg: float,
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    #计算得分 shape = (num_train, num_class)
    scores = X.mm(W)
    #计算正确得分，这里把本来为1行num_train列化为num_train行1列为了方便下面计算
    correct_scores = scores[range(num_train), y].reshape(-1, 1)
    zeros = torch.zeros_like(scores)
    #计算间隔，scores.shape = (num_train, num_class), 而正确类别得分是(num_train, 1),相减时通过广播机制自动扩充至(num_train, num_class)
    #把当前num_train的正确分数复制num_class份再相减(横向复制), maximum和0矩阵逐个元素比较，大者胜出
    margin = torch.maximum(scores - correct_scores + 1, zeros)
    #除去正确类别影响，如果不除去的话对于正确类别scores - correct_scores = 0，结果为1会影响计算
    margin[range(num_train), y] = 0
    loss = margin.sum() / num_train + reg * torch.sum(W * W)

    #把间隔大于0标记为1
    margin[margin > 0] = 1
    margin[margin < 0] = 0
    #当前margin的正确类别抵消了多少个错误类比
    margin[range(num_train), y] = -torch.sum(margin, dim=1)

    #计算梯度得到dW.shape = (dim, Class)
    dW = torch.mm(X.T, margin) / num_train + 2 * reg * W

    return loss, dW

def softmax_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float
):
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    scores = torch.mm(X, W)
    #通过keepdim = True 让其保持 m行 x 1列的形式，下softmax同, (max会返回值和索引，只需要值)
    scores_stable = scores - scores.max(dim=1, keepdim=True).values
    correct_scores = scores_stable[range(num_train), y]
    # e ^ (scores)
    #计算某个样本在所有种类的E次方, 得到(num_train, )
    exp_sum = torch.sum(torch.exp(scores_stable), dim=1)
    #loss = -log( E^正确类别得分 / E^所有种类得分 )
    loss = -torch.log(torch.exp(correct_scores) / exp_sum)
    loss = loss.sum()
    loss = loss / num_train + reg * torch.sum(W * W)

    #计算梯度，把正确类别置为-1，和某个样本的 E^每一个类别对应的得分 / E^每个类别之和
    correct_matrix = torch.zeros_like(scores)
    correct_matrix[range(num_train), y] = -1
    dW = torch.mm(X.t(), torch.exp(scores_stable) / exp_sum.reshape(-1, 1) + correct_matrix)
    dW = dW / num_train + 2 * reg * W

    return loss, dW

def sample_batch(
        X: torch.Tensor,
        y: torch.Tensor,
        num_train: int,
        batch_size: int
):
    random_samples = torch.randint(0, num_train, (batch_size, ))
    X_batch = X[random_samples]
    y_batch = y[random_samples]

    return X_batch, y_batch

def train_linear_classifier(
        loss_func: Callable,
        W: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
):
    num_train, dim = X.shape
    if W is None:
        num_class = y.max() + 1
        W = torch.randn(dim, num_class, device=X.device, dtype=X.dtype) * 0.000001
    else:
        num_class = W.shape[1]

    loss_history = []
    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())
        # 更新权重矩阵
        W -= learning_rate * grad

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history

"""
    计算得分，同时获得当前类别在所有类别中最大的那一个的索引，即为预测标签
    y_pred = torch.zeros(X.shape[0])    要对当前输入进来的X的每一个都预测即 = num_train
"""
def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    scores = torch.mm(X, W)
    y_pred = torch.argmax(scores, dim=1)

    return y_pred


# Template class modules
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):

        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)

"""
    设定超参数
"""
def svm_get_search_params():

    learning_rates = []
    regularization_strengths = []
    #见论文：Cyclical Learning Rates for Training Neural Networks
    learning_rates = [1e-4, 1e-3, 5e-2, 2e-2, 1e-2]
    regularization_strengths = [1e-3, 1e-2, 1e-1, 1e0, 1e1]


    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    learning_rate: float,
    reg: float,
    num_iters: int = 2000,
):
    """
        训练一个 LinearClassifier 实例同时返回学习完成的实例在训练集和验证集上的准确率
    :param cls: 线性分类器实例 - LinearSVM() 和 Softmax()
    :param data_dict: 键值对 ['X_train', 'y_train', 'X_val', 'y_val']作为键值对的key，值为对应的张量
    :param learning_rate:
    :param reg:
    :param num_iters:   当你不确定验证是否正确时可以设置较小的 num_iters 确定正确后再换位最终的
    :return:    cls: 经过 num_iters 次迭代的 LinearClassifier 实例
                train_acc, val_acc: 准确率
    """
    #获取训练集，及对应训练标签，验证集，即对应验证标签
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    #调用子类的默认损失函数
    cls.train(X_train, y_train, learning_rate, reg, num_iters, batch_size=200, verbose=False)
    #求准确率
    y_train_pred = cls.predict(X_train)
    train_acc = (y_train == y_train_pred).double().mean().item() * 100.0

    y_val_pred = cls.predict(X_val)
    val_acc = (y_val == y_val_pred).double().mean().item() * 100.0

    return cls, train_acc, val_acc

def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    learning_rates = [1e-4, 1e-3, 5e-2, 2e-2, 1e-2]
    regularization_strengths = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

    return learning_rates, regularization_strengths