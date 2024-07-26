import torch
from a3_helper import softmax_loss
from eecs598 import Solver

class Linear(object):

    # @staticmethod: 表示方法是一个静态方法，不需要类实例即可调用
    @staticmethod
    def forward(x, w, b):
        out = None
        num_input = x.shape[0]
        out = torch.mm(x.reshape(num_input, -1), w) + b

        cache = (x, w, b)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        :param dout: 上游导数 (N, M) N - 批次中的样本数量，M是输出维度
        :param cache: 前向传播中的数据
        :return: 输入数据的梯度dx, 权重的梯度dw, 偏置的梯度db
        """
        x, w, b = cache
        dx, dw, db = None, None, None

        num_input = x.shape[0]
        x_flat = x.reshape(num_input, -1)

        # 输入数据的梯度(本层梯度) = 上游导数dout * 权重转置w.t()
        dx = torch.mm(dout, w.t()).reshape(x.shape)
        # 权重梯度 = 展平后的x矩阵.t() * 上游导数dout
        dw = torch.mm(x_flat.t(), dout)
        # 上游导数dim = 0求和
        db = torch.sum(dout, dim=0)

        return dx, dw, db

class ReLU(object):

    @staticmethod
    def forward(x):
        out = None
        # out = x * (x > 0)
        out = torch.relu(x)
        cache = x

        return out, cache

    @staticmethod
    def backward(dout, cache):
        dx, x = None, cache

        zeros = torch.zeros_like(dout)
        dx = dout * (x > 0)

        return dx

class Linear_ReLU(object):
    @staticmethod
    def forward(x, w, b):
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)

        return out, cache
    @staticmethod
    def backward(dout, cache):
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)

        return dx, dw, db

class TwoLayerNet(object):

    def __init__(
        self,
        input_dim=3*32*32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=torch.float32,
        device='cpu'
    ):
        self.params = {}
        self.reg = reg

        #First Layer
        self.params['W1'] = torch.normal(0.0, weight_scale, size=(input_dim, hidden_dim), dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        # Second Layer
        self.params['W2'] = torch.normal(0.0, weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))
    def loss(self, X, y = None):
        scores = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        out1, cache1 = Linear_ReLU.forward(X, W1, b1)
        scores, cache2= Linear.forward(out1, W2, b2)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscore = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        # backward pass
        dh, dW2, db2 = Linear.backward(dscore, cache2)
        dx, dW1, db1 = Linear_ReLU.backward(dh, cache1)
        grads['W1'] = dW1 + 2 * self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + 2 * self.reg * W2
        grads['b2'] = db2

        return loss, grads