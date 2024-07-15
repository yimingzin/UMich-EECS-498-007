import torch
from typing import Dict, List


# num_train = 500, num_test = 250
def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    :param x_train: Tensor of shape (num_train, D1, D2, ...)    :param x_test: Tensor of shape (num_test, D1, D2, ...)    :return: distances: Tensor of shape (num_train, num_test)
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    distances = torch.zeros(num_train, num_test)

    # 假设我有500张训练图片，250张测试图片，两层循环求每 1 个图片对 250张测试图片的Euclidean Distance
    for i in range(num_train):
        for j in range(num_test):
            # 欧几里得距离公式 - 这里的i代表第i张训练样本所有的数值，j代表第j张测试样本的数值
            distances[i, j] = torch.sum((x_train[i] - x_test[j])**2).pow(1/2)

    return distances


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    distances = torch.zeros(num_train, num_test)

    # 把训练集和测试集铺平 成 (num_train, 3072) 和 (num_test, 3072)型
    x_train_temp = x_train.reshape(num_train, -1)
    x_test_temp = x_test.reshape(num_test, -1)

    # 500张训练图片，取出1张，复制 250张(num_test)份, 直接求他们直接的距离
    for i in range(num_train):
        # x_train[i] 是 1 x 3072, x_test_temp是 num_test x 3072, 通过广播机制把x_train复制num_test份
        # 我们求和是对测试样本的所有像素点求和，即3072
        distances[i] = torch.sum((x_train_temp[i] - x_test_temp)**2, dim=1).pow(1/2)

    return distances


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    distance = torch.zeros(num_train, num_test)

    # 把训练集和测试集铺平 成 (num_train, 3072) 和 (num_test, 3072)型
    train_temp = x_train.reshape(num_train, -1)
    test_temp = x_test.reshape(num_test, -1)

    # test_temp_square对dim=1求和后的形状为x_sum = [10, 26, 42], 通过保持维度保持为x_sum = [[10], [26], [42]]
    train_temp_square = torch.sum(train_temp ** 2, dim=1, keepdim=True)
    test_temp_square = torch.sum(test_temp ** 2, dim=1)
    # 广播机制两边都复制到相同的份数相加
    train_test_inner = torch.mm(train_temp, test_temp.t()) * 2

    distance = (train_temp_square + test_temp_square - train_test_inner).pow(1 / 2)

    return distance


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    num_train, num_test = dists.shape
    # 我们是去预测未知样本的类别(标签),所以这里是测试样本的数量
    y_pred = torch.zeros(num_test, dtype=torch.float64)

    # 从列出发(测试样本), 找距离最小的k个值的索引
    top_k = torch.topk(dists, k, 0, False).indices
    for j in range(num_test):
        # top_k[:, j] 包含的是第 j 个测试样本的 k 个最近训练样本的索引,torch.mode会返回众数本身和其索引，只需要本身
        y_pred[j] = torch.mode(y_train[top_k[:, j]])[0].item()

    return y_pred


class KnnClassifier:
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: torch.Tensor, k: int = 1):
        y_test_pred = None
        distances = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(distances, self.y_train, k)

        return y_test_pred

    def check_accuracy(
            self,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
            k: int = 1,
            quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test        data. Returns the accuracy of the classifier on the test data, and        also prints a message giving the accuracy.
        Args:            x_test: Tensor of shape (num_test, C, H, W) giving test samples.            y_test: int64 Tensor of shape (num_test,) giving test labels.            k: The number of neighbors to use for prediction.            quiet: If True, don't print a message.
        Returns:            accuracy: Accuracy of this classifier on the test data, as a                percent. Python float in the range [0, 100]        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        num_folds: int = 5,
        k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    x_train_folds = []
    y_train_folds = []
    """  
        x_train_folds.shape = [(100, 3072), (100, 3072), (100, 3072), (100, 3072)]        y_train_folds.shape = [(100, ), (100, ), (100, ), (100, ), (100, ), ]    """
    x_train_folds = x_train.chunk(num_folds)
    y_train_folds = y_train.chunk(num_folds)

    k_to_accuracies = {}  # {k1: [], k2: [], ....}
    # 对于每个k都有num_folds个 accuracy
    for k in k_choices:
        k_to_accuracies[k] = []
        for i in range(num_folds):
            # 选取训练样本中的第i个作为验证集，其余的作为测试集 (标签同理)
            x_val = x_train_folds[i]
            y_val = y_train_folds[i]
            """  
            假设我们有5个折，每个折表示为 x_train_folds 列表中的一个元素：x_train_folds = [fold0, fold1, fold2, fold3, fold4]  
            如果 i=2，那么我们希望把 fold2 作为验证集，其他的折作为训练集。  
            x_train_folds[:2] 得到 [fold0, fold1]            x_train_folds[3:] 得到 [fold3, fold4]            """
            x_train = torch.cat(x_train_folds[:i] + x_train_folds[i + 1:], dim=0)
            y_train = torch.cat(y_train_folds[: i] + y_train_folds[i + 1:], dim=0)

            knn_classifier = KnnClassifier(x_train, y_train)
            i_acc = knn_classifier.check_accuracy(x_val, y_val, k=k, quiet=True)
            k_to_accuracies[k].append(i_acc)

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    best_k = 0
    # 匿名函数，它接受参数k，返回 k_to_accuracies[k] 列表的总和
    func = lambda k: sum(k_to_accuracies[k])
    # 按照总和找到最大的K值
    best_k = max(k_to_accuracies, key=func)

    return best_k
