import torch
from a3_helper import softmax_loss
from eecs598 import Solver

class Linear(object):
    @staticmethod
    def forward(x, w, b):
        num_input = x.shape[0]
        out = torch.mm(x.reshape(num_input, -1), w) + b
        cache = (x, w, b)
        return out, cache
    @staticmethod
    def backward(dout, cache):
        dx, dw, db = None, None, None
        x, w, b = cache
        num_input = x.shape[0]
        x_flat = x.reshape(num_input, -1)

        dx = torch.mm(dout, w.t()).reshape(x.shape)
        dw = torch.mm(x_flat.t(), dout)
        db = torch.sum(dout, dim=0)

        return dx, dw, db
class ReLU(object):
    @staticmethod
    def forward(x):
        out = x * (x > 0)
        cache = x
        return out, cache
    @staticmethod
    def backward(dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx

class Linear_ReLU(object):
    @staticmethod
    def forward(x, w, b):
        out_linear, cache_linear = Linear.forward(x, w, b)
        out, cache_relu = ReLU.forward(out_linear)
        cache = (cache_linear, cache_relu)
        return out, cache
    @staticmethod
    def backward(dout, cache):
        cache_linear, cache_relu = cache
        da = ReLU.backward(dout, cache_relu)
        dx, dw, db = Linear.backward(da, cache_linear)
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

        self.params['W1'] = torch.normal(mean=0, std=weight_scale, size=(input_dim, hidden_dim), dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = torch.normal(mean=0, std=weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Save in {}".format(path))

    def load(self, path, dtype, device):
        # map_location='cpu' 确保无论原来是用什么设备保存的，都先加载到 CPU 上，之后可以根据需要再移动到指定设备
        checkpoint = torch.load(path, map_location='cpu')
        self.params['reg'] = checkpoint['reg']
        self.params['params'] = checkpoint['params']

        print("load checkpoint file:{}".format(path))
    """
        示例使用
        model = TwoLayerNet(reg=0.1)
        model.save('model.pth')

        loaded_model = TwoLayerNet()
        loaded_model.load('model.pth')
        print("Loaded reg:", loaded_model.reg)
    """
    def loss(self, X, y = None):
        scores = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 前向传播
        hidden, cache_one = Linear_ReLU.forward(X, W1, b1)
        scores, cache_two = Linear.forward(hidden, W2, b2)

        if y is None:
            return scores
        loss, grad = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
        # 反向传播
        dhidden, dW2, db2 = Linear.backward(dscores, cache_two)
        dX, dW1, db1 = Linear_ReLU.backward(dhidden, cache_one)
        grad['W1'] = dW1 + 2 * self.reg * W1
        grad['b1'] = db1
        grad['W2'] = dW2 + 2 * self.reg * W2
        grad['b2'] = db2

        return loss, grad