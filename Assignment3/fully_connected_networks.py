import torch
from a3_helper import softmax_loss
from eecs598 import Solver


class Linear(object):
    @staticmethod
    def forward(x, w, b):
        num_input = x.shape[0]
        x_flat = x.reshape(num_input, -1)

        out = torch.mm(x_flat, w) + b
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
            input_dim=3 * 32 * 32,
            hidden_dim=100,
            num_classes=10,
            reg=0.0,
            weight_scale=1e-3,
            dtype=torch.float32,
            device='cpu'
    ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = torch.normal(mean=0, std=weight_scale, size=(input_dim, hidden_dim), dtype=dtype,
                                         device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = torch.normal(mean=0, std=weight_scale, size=(hidden_dim, num_classes), dtype=dtype,
                                         device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'params': self.params,
            'reg': self.reg
        }
        torch.save(checkpoint, path)
        print("checkpoint saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint in {}".format(path))

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        num_input, input_dim = X.shape

        out_one, cache_one = Linear_ReLU.forward(X, W1, b1)
        scores, cache_two = Linear.forward(out_one, W2, b2)

        if y is None:
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        dhidden, dW2, db2 = Linear.backward(dscores, cache_two)
        dx, dW1, db1 = Linear_ReLU.backward(dhidden, cache_one)

        grads['W1'] = dW1 + 2 * self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + 2 * self.reg * W2
        grads['b2'] = db2

        return loss, grads


class FullyConnectedNet(object):
    def __init__(
            self,
            hidden_dim,
            input_dim=3 * 32 * 32,
            num_classes=10,
            reg=0.0,
            weight_scale=1e-2,
            dropout=0.0,
            seed=None,
            dtype=torch.float32,
            device='cpu'
    ):
        self.params = {}
        self.num_layers = len(hidden_dim) + 1
        self.reg = reg
        self.use_dropout = dropout != 0
        self.dtype = dtype

        self.params['W1'] = torch.normal(mean=0, std=weight_scale, size=(input_dim, hidden_dim[0]), dtype=dtype,
                                         device=device)
        self.params['b1'] = torch.zeros(hidden_dim[0], dtype=dtype, device=device)

        L = self.num_layers
        for i in range(2, L):
            self.params[f'W{i}'] = torch.normal(mean=0, std=weight_scale, size=(hidden_dim[i - 2], hidden_dim[i - 1]),
                                                dtype=dtype, device=device)
            self.params[f'b{i}'] = torch.zeros(hidden_dim[i - 1], dtype=dtype, device=device)
        # 最后的W应为最后一个隐藏层神经元数量到输出层, 输出层是num_classes，L是总层数，L-1是hidden_dim的长度, L-2是最后一个元素的索引
        self.params[f'W{L}'] = torch.normal(mean=0, std=weight_scale, size=(hidden_dim[L - 2], num_classes),
                                            dtype=dtype, device=device)
        self.params[f'b{L}'] = torch.zeros(num_classes, dtype=dtype, device=device)

        self.dropout_params = {}
        if self.use_dropout:
            self.dropout_params = {
                'mode': 'train',
                'p': dropout
            }
            if seed is not None:
                self.dropout_params['seed'] = seed

    def save(self, path):
        checkpoint = {
            'params': self.params,
            'num_layers': self.num_layers,
            'reg': self.reg,
            'use_dropout': self.use_dropout,
            'dtype': self.dtype,
            'dropout_params': self.dropout_params
        }
        torch.save(checkpoint, path)
        print("checkpoint saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.num_layers = checkpoint['num_layers']
        self.reg = checkpoint['reg']
        self.use_dropout = checkpoint['use_dropout']
        self.dtype = dtype
        self.dropout_params = checkpoint['dropout_params']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("checkpoint saved in {}".format(path))

    def loss(self, X, y=None):
        X = X.to(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'
        if self.use_dropout:
            self.dropout_params['mode'] = mode

        scores = None
        # forward
        L = self.num_layers
        cache = []
        h, cache_one = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        cache.append(cache_one)

        for i in range(2, L):
            h, cache_i = Linear_ReLU.forward(h, self.params[f'W{i}'], self.params[f'b{i}'])
            cache.append(cache_i)
        # 这里是要乘最后一层，L代表网络的总层数
        scores, cache_fin = Linear.forward(h, self.params[f'W{L}'], self.params[f'b{L}'])
        cache.append(cache_fin)

        if mode == 'test':
            return scores

        # loss
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        for i in range(L):
            loss += self.reg * torch.sum(self.params[f'W{i + 1}'] ** 2)

        # grads (backward)
        dh, grads[f'W{L}'], grads[f'b{L}'] = Linear.backward(dscores, cache[-1])
        grads[f'W{L}'] += 2 * self.reg * self.params[f'W{L}']

        for i in range(L - 1, 0, -1):
            # cache是列表，存储了每一层前向传播的中间结果, 第一层的中间结果存储在 cache[0], 第二层的中间结果存储在 cache[1]
            dh, grads[f'W{i}'], grads[f'b{i}'] = Linear_ReLU.backward(dh, cache[i - 1])
            grads[f'W{i}'] += 2 * self.reg * self.params[f'W{i}']

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    solver = solver = Solver(model, data_dict, optim_config={'learning_rate': 5e-2, },
                             lr_decay=0.95,
                             num_epochs=20, batch_size=100,
                             print_every=100,
                             device='cuda'
                             )
    return solver


def get_three_layer_network_params():
    weight_scale = 5e-1
    learning_rate = 1e-2
    return weight_scale, learning_rate


def get_five_layer_network_params():
    learning_rate = 2e-1
    weight_scale = 1e-1
    return weight_scale, learning_rate
