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
        out_relu, cache_relu = ReLU.forward(out_linear)

        cache = (cache_linear, cache_relu)
        return out_relu, cache

    @staticmethod
    def backward(dout, cache):
        cache_linear, cache_relu = cache
        da = ReLU.backward(dout, cache_relu)
        dx, dw, db = Linear.backward(da, cache_linear)

        return dx, dw, db


class Dropout(object):
    @staticmethod
    def forward(x, dropout_params):
        mode, p = dropout_params['mode'], dropout_params['p']

        if 'seed' in dropout_params:
            torch.manual_seed(dropout_params['seed'])

        out = None
        mask = (torch.rand(x.shape) > p) / (1 - p)
        mask = mask.cuda()

        if mode == 'train':
            out = x * mask
        elif mode == 'test':
            out = x

        cache = (mask, dropout_params)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        mask, drop_params = cache
        mode = drop_params['mode']
        dx = None

        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout

        return dx


class TwoLayerNet(object):
    def __init__(
            self,
            input_dim=3 * 32 * 32,
            hidden_dim=100,
            num_classes=10,
            reg=0.0,
            weight_scale=1e-2,
            dtype=torch.float32,
            device='cpu'
    ):
        self.params = {}
        self.reg = reg

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

        out_1, cache_1 = Linear_ReLU.forward(X, W1, b1)
        scores, cache_2 = Linear.forward(out_1, W2, b2)

        if y is None:
            return scores
        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        dh, grads['W2'], grads['b2'] = Linear.backward(dscores, cache_2)
        grads['W2'] += 2 * self.reg * self.params['W2']

        dx, grads['W1'], grads['b1'] = Linear_ReLU.backward(dh, cache_1)
        grads['W1'] += 2 * self.reg * self.params['W1']

        return loss, grads


class FullyConnectedNet(object):
    def __init__(
            self,
            hidden_dims,
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
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        self.use_dropout = dropout != 0.0
        self.dtype = dtype
        self.dropout_params = {}

        L = self.num_layers
        self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(input_dim, hidden_dims[0]), dtype=dtype,
                                         device=device)
        self.params['b1'] = torch.zeros(hidden_dims[0], dtype=dtype, device=device)

        for i in range(2, L):
            self.params[f'W{i}'] = torch.normal(mean=0.0, std=weight_scale,
                                                size=(hidden_dims[i - 2], hidden_dims[i - 1]), dtype=dtype,
                                                device=device)
            self.params[f'b{i}'] = torch.zeros(hidden_dims[i - 1], dtype=dtype, device=device)

        self.params[f'W{L}'] = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dims[-1], num_classes),
                                            dtype=dtype, device=device)
        self.params[f'b{L}'] = torch.zeros(num_classes, dtype=dtype, device=device)

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
            'reg': self.reg,
            'num_layers': self.num_layers,
            'use_dropout': self.use_dropout,
            'dtype': self.dtype,
            'dropout_params': self.dropout_params
        }
        torch.save(checkpoint, path)
        print("checkpoint saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dtype = dtype
        self.dropout_params = checkpoint['dropout_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint in {}".format(path))

    def loss(self, X, y=None):
        X = X.to(self.dtype)

        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        if self.use_dropout:
            self.dropout_params['mode'] = mode

        L = self.num_layers
        scores, cache, cache_dropout = None, [], []

        scores, cache_1 = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        cache.append(cache_1)
        if self.use_dropout:
            scores, cache_1 = Dropout.forward(X, self.dropout_params)
            cache_dropout.append(cache_1)

        for i in range(2, L):
            scores, cache_i = Linear_ReLU.forward(scores, self.params[f'W{i}'], self.params[f'b{i}'])
            cache.append(cache_i)
            if self.use_dropout:
                scores, cache_i = Dropout.forward(scores, self.dropout_params)
                cache_dropout.append(cache_i)

        scores, cache_fin = Linear.forward(scores, self.params[f'W{L}'], self.params[f'b{L}'])
        cache.append(cache_fin)

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)
        for i in range(L):
            loss += self.reg * torch.sum(self.params[f'W{i + 1}'] ** 2)

        dh, grads[f'W{L}'], grads[f'b{L}'] = Linear.backward(dscores, cache_fin)
        grads[f'W{L}'] += 2 * self.reg * self.params[f'W{L}']

        for i in range(L - 1, 0, -1):
            if self.use_dropout:
                dh = Dropout.backward(dh, cache_dropout[i - 1])
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


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    w = w - config['learning_rate'] * dw

    return w


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('v', 0.9)
    config.setdefault('velocity', torch.zeros_like(w))

    v = config['velocity'] * config['v'] - config['learning_rate'] * dw
    next_w = w + v

    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('r', 0.99)
    config.setdefault('cache', torch.zeros_like(w))
    config.setdefault('epsilon', 1e-8)

    r = config['r'] * config['cache'] + (1 - config['r']) * (dw ** 2)
    next_w = w - (config['learning_rate'] * dw) / (torch.sqrt(r) + config['epsilon'])

    config['cache'] = r
    return next_w, config


def adam(w, dw, config = None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta_1', 0.9)
    config.setdefault('beta_2', 0.999)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('epsilon', 1e-8)
    config.setdefault('t', 0)

    config['t'] += 1

    m = config['beta_1'] * config['m'] + (1 - config['beta_1']) * dw
    m_hat = m / (1 - config['beta_1'] ** config['t'])

    v = config['beta_2'] * config['v'] + (1 - config['beta_2']) * (dw ** 2)
    v_hat = v / (1 - config['beta_2'] ** config['t'])

    next_w = w - (config['learning_rate'] * m_hat) / (torch.sqrt(v_hat) + config['epsilon'])

    config['m'], config['v'] = m, v
    return next_w, config


# extra : nesterov_momentum
def nesterov_momentum(w, dw, config = None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    config.setdefault('velocity', torch.zeros_like(w))

    #提前更新速度
    prev_v = config['velocity']
    v = config['momentum'] * prev_v - config['learning_rate'] * dw

    # 使用 Nesterov 动量更新权重
    next_w = w + config['momentum'] * v - config['learning_rate'] * dw

    config['velocity'] = v

    return next_w, config
