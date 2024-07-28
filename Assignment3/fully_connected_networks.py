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
            input_dim=3 * 32 * 32,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
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
            'reg': self.reg,
        }
        torch.save(checkpoint, path)
        print("checkpoint saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y = None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # forward
        out_one, cache_one= Linear_ReLU.forward(X, W1, b1)
        scores, cache_two = Linear.forward(out_one, W2, b2)
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        # backward
        dhidden, dw2, db2 = Linear.backward(dscores, cache_two)
        dx, dw1, db1 = Linear_ReLU.backward(dhidden, cache_one)

        grads['W1'] = dw1 + 2 * self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dw2 + 2 * self.reg * W2
        grads['b2'] = db2

        return loss, grads


#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    solver = solver = Solver(model, data_dict, optim_config={'learning_rate': 5e-2,},
                             lr_decay=0.95,
                             num_epochs=20, batch_size=100,
                             print_every=100,
                             device='cuda'
                        )
    return solver


class FullyConnectedNet(object):
    def __init__(
        self,
        hidden_dim,
        input_dim = 3*32*32,
        num_classes = 10,
        reg = 0.0,
        weight_scale = 1e-2,
        dropout = 0.0,
        seed = None,
        dtype = torch.float,
        device = 'cpu'
    ):
        self.num_layers = len(hidden_dim) + 1
        self.reg = reg
        self.use_dropout = dropout != 0.0
        self.dtype = dtype
        self.params = {}

        L = self.num_layers
        self.params['W1'] = torch.normal(mean=0, std=weight_scale, size=(input_dim, hidden_dim[0]), dtype=dtype,device=device)
        self.params['b1'] = torch.zeros(hidden_dim[0], dtype=dtype, device=device)

        for i in range(2, L):
            self.params[f'W{i}'] = torch.normal(mean=0, std=weight_scale, size=(hidden_dim[i - 2], hidden_dim[i - 1]),dtype=dtype, device=device)
            self.params[f'b{i}'] = torch.zeros(hidden_dim[i - 1], dtype=dtype, device=device)

        self.params[f'W{L}'] = torch.normal(mean=0, std=weight_scale, size=(hidden_dim[L - 2], num_classes), dtype=dtype,device=device)
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
            'reg': self.reg,
            'num_layers': self.num_layers,
            'params': self.params,
            'use_dropout': self.use_dropout,
            'dropout_params': self.dropout_params,
            'dtype': self.dtype
        }
        torch.save(checkpoint, path)
        print("checkpoint saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.params = checkpoint['params']
        self.use_dropout = checkpoint['use_dropout']
        self.dtype = dtype
        self.dropout_params = checkpoint['dropout_params']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint in {}".format(path))

    def loss(self, X, y = None):
        X = X.to(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        if self.use_dropout:
            self.dropout_params['mode'] = mode
        scores = None
        L = self.num_layers
        cache = []

        h, cache_0 = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        cache.append(cache_0)
        for i in range(2, L):
            h, cache_i = Linear_ReLU.forward(h, self.params[f'W{i}'], self.params[f'b{i}'])
            cache.append(cache_i)
        scores, cache_final = Linear.forward(h, self.params[f'W{L}'], self.params[f'b{L}'])
        cache.append(cache_final)

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        for i in range(L):
            loss += self.reg * torch.sum(self.params[f'W{i+1}'] ** 2)
        # dh, grads[f'W{L}'], grads[f'b{L}'] = Linear.backward(dscores, cache[L-1])
        dh, grads[f'W{L}'], grads[f'b{L}'] = Linear.backward(dscores, cache[-1])
        grads[f'W{L}'] += 2 * self.reg * self.params[f'W{L}']

        for i in range(L-1, 0, -1):
            dh, grads[f'W{i}'], grads[f'b{i}'] = Linear_ReLU.backward(dh, cache[i-1])
            grads[f'W{i}'] += 2 * self.reg * self.params[f'W{i}']

        return loss, grads
