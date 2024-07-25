import torch
import statistics
import random
from linear_classifier import sample_batch
from typing import Dict, List, Callable, Optional

# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))

def nn_forward_pass(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor
):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    num_input, dim = X.shape

    # 计算输入层到隐藏层，
    # 1.进行线性变换hidden_layer=X ⋅ W1 + b1,
    # 2.应用ReLU激活函数hidden = ReLU(hidden_layer)
    h1 = torch.mm(X, W1) + b1
    zeros = torch.zeros_like(h1)
    hidden = torch.maximum(h1, zeros)

    # hidden = torch.relu(h1)等价写法
    # 对隐藏层到输出层进行线性变换
    scores = torch.mm(hidden, W2) + b2

    return scores, hidden

def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    num_input, dim = X.shape

    scores, hidden= nn_forward_pass(params, X)
    if y is None:
        return scores
    # 保持数字稳定性
    scores -= scores.max(dim=1, keepdim=True).values
    correct_scores = scores[range(num_input), y]
    sum_exp = torch.sum(torch.exp(scores), dim=1)
    # loss计算参考softmax的计算
    loss = -torch.log(torch.exp(correct_scores) / sum_exp)
    loss = loss.sum() / num_input + 0.5 * reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

    grads = {}
    grad_scores = torch.exp(scores) / sum_exp.reshape(-1, 1)
    grad_scores[range(num_input), y] -= 1
    grad_scores /= num_input

    grads['W2'] = torch.mm(hidden.t(), grad_scores) + reg * W2
    grads['b2'] = torch.sum(grad_scores, dim=0)

    grad_hidden = torch.mm(grad_scores, W2.t())
    grad_hidden[hidden <= 0] = 0

    grads['W1'] = torch.mm(X.t(), grad_hidden) + reg * W1
    grads['b1'] = torch.sum(grad_hidden, dim=0)

    return loss, grads

def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):

    y_pred = None
    scores = loss_func(params, X)
    # 求类别分数最大的索引
    y_pred = torch.argmax(scores, dim=1)

    return y_pred

def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        params['W1'] -= learning_rate * grads['W1']
        params['b1'] -= learning_rate * grads['b1']
        params['W2'] -= learning_rate * grads['W2']
        params['b2'] -= learning_rate * grads['b2']

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }

def nn_get_search_params():
    learning_rates = [1e-4, 1e-3, 1e-2, 1e0]
    hidden_sizes = [8, 32, 128]
    regularization_strengths = [1e-3, 1e-1, 1e1]
    learning_rate_decays = [1.0, 0.95, 0.90]

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays
    )

def find_best_net(
        data_dict: Dict[str, torch.Tensor],
        get_param_set_fn: Callable
):
    # 在训练过程中, 如果当前模型在验证集上的准确率超过了best_val_acc就更新best_net为当前模型
    best_net = None
    # 在训练过程中, 如果当前模型在验证集上的准确率超过best_val_acc就更新best_stat为当前模型的训练统计数据
    best_stat = None
    best_val_acc = 0.0
    learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = get_param_set_fn()

    dim = data_dict['X_train'].shape[1]
    num_class = max(data_dict['y_train']) + 1

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for reg in regularization_strengths:
                for lr_decay in learning_rate_decays:
                    model = TwoLayerNet(dim, hidden_size, num_class)
                    train_stats = model.train(
                        data_dict['X_train'],
                        data_dict['y_train'],
                        data_dict['X_val'],
                        data_dict['y_val'],
                        lr,
                        lr_decay,
                        reg,
                        1000,
                        1000
                    )
                    y_val_preds = model.predict(data_dict['X_val'])
                    val_acc = (y_val_preds == data_dict['y_val']).double().mean().item()
                    if val_acc > best_val_acc:
                        best_net, best_stat, best_val_acc = model, train_stats, val_acc
    return best_net, best_stat, best_val_acc