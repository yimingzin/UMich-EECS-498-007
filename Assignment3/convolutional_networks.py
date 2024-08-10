import time

import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear, Linear_ReLU, Solver, adam, ReLU, sgd_momentum, sgd


class Conv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        # 假设输入数据 x 的形状是 (N, 3, 32, 32)，表示 N 张 32x32 的 RGB 图像
        N, C, H, W = x.shape
        # 假设有10个滤波器，每个滤波器的形状是(3, 5, 5)，表示每个滤波器跨越3个通道(RGB), 并且滤波器大小为 5x5
        F, C, HH, WW = w.shape
        # 步长 和 0填充
        stride, pad = conv_param['stride'], conv_param['pad']

        # 计算输出数据的高度H' 和宽度W' (注意需确保为整数)
        H_prime = 1 + (H - HH + 2 * pad) // stride
        W_prime = 1 + (W - WW + 2 * pad) // stride

        # 对输入数据进行0填充
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # 初始化输出数据
        out = torch.zeros((N, F, H_prime, W_prime), dtype=x.dtype, device=x.device)

        # 图片索引
        for n in range(N):
            # 卷积核索引
            for f in range(F):
                # 当前图片在对应卷积核，滤波器在图像上滑动，每次移动stride个像素,覆盖一个与滤波器大小相同的感受野,进行卷积计算
                for h_prime in range(H_prime):
                    for w_prime in range(W_prime):
                        h_start, h_end = h_prime * stride, h_prime * stride + HH
                        w_start, w_end = w_prime * stride, w_prime * stride + WW
                        receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]

                        out[n, f, h_prime, w_prime] = torch.sum(receptive_field * w[f]) + b[f]

        cache = (x, w, b, conv_param)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        # unpack cache
        x, w, b, conv_param = cache
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        N, F, H_prime, W_prime = dout.shape

        # initialize
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)

        # pad
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        dx_padded = torch.nn.functional.pad(dx, (pad, pad, pad, pad))

        for n in range(N):
            for f in range(F):
                for h_prime in range(H_prime):
                    for w_prime in range(W_prime):
                        h_start, h_end = h_prime * stride, h_prime * stride + HH
                        w_start, w_end = w_prime * stride, w_prime * stride + WW
                        receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]

                        dx_padded[n, :, h_start:h_end, w_start:w_end] += dout[n, f, h_prime, w_prime] * w[f]
                        dw[f] += receptive_field * dout[n, f, h_prime, w_prime]
                        db[f] += dout[n, f, h_prime, w_prime]

        # unpad data
        # 从第 pad 个元素开始, 到倒数第 pad 个元素结束: 去掉前 pad 个和后 pad 个元素
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
        return dx, dw, db


class MaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']

        H_prime = 1 + (H - pool_height) // stride
        W_prime = 1 + (W - pool_width) // stride

        out = torch.zeros((N, C, H_prime, W_prime), dtype=x.dtype, device=x.device)

        for n in range(N):
            for c in range(C):
                for h_prime in range(H_prime):
                    for w_prime in range(W_prime):
                        h_start, h_end = h_prime * stride, h_prime * stride + pool_height
                        w_start, w_end = w_prime * stride, w_prime * stride + pool_width
                        receptive_field = x[n, c, h_start:h_end, w_start:w_end]

                        out[n, c, h_prime, w_prime] = torch.max(receptive_field)

        cache = x, pool_param
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, pool_param = cache
        N, C, H, W = x.shape
        N, C, H_prime, W_prime = dout.shape
        pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']

        dx = torch.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h_prime in range(H_prime):
                    for w_prime in range(W_prime):
                        h_start, h_end = h_prime * stride, h_prime * stride + pool_height
                        w_start, w_end = w_prime * stride, w_prime * stride + pool_width
                        receptive_field = x[n, c, h_start:h_end, w_start:w_end]
                        # 创建掩码确定池化区域最大值的位置
                        mask = (receptive_field == torch.max(receptive_field))
                        # 把上游梯度累加到确定的最大值位置
                        dx[n, c, h_start:h_end, w_start:w_end][mask] += dout[n, c, h_prime, w_prime]
        return dx


class FastConv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']

        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)

        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)

        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, _, _, _, tx, out, layer = cache
        try:
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight)
            db = torch.zeros_like(layer.bias)

        return dx, dw, db


class FastMaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']

        layer = torch.nn.MaxPool2d((pool_height, pool_width), stride=stride)

        tx = x.detach()
        tx.requires_grad = True

        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, _, tx, out, layer = cache
        try:
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)

        return dx


class Conv_ReLU(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        out_conv, cache_conv = FastConv.forward(x, w, b, conv_param)
        out_relu, cache_relu = ReLU.forward(out_conv)

        cache = (cache_conv, cache_relu)

        return out_relu, cache

    @staticmethod
    def backward(dout, cache):
        cache_conv, cache_relu = cache
        da = ReLU.backward(dout, cache_relu)
        dx, dw, db = FastConv.backward(da, cache_conv)

        return dx, dw, db


class Conv_ReLU_Pool(object):
    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        out_conv, cache_conv = FastConv.forward(x, w, b, conv_param)
        out_relu, cache_relu = ReLU.forward(out_conv)
        out_pool, cache_pool = FastMaxPool.forward(out_relu, pool_param)

        cache = (cache_conv, cache_relu, cache_pool)
        return out_pool, cache

    @staticmethod
    def backward(dout, cache):
        cache_conv, cache_relu, cache_pool = cache
        ds = FastMaxPool.backward(dout, cache_pool)
        da = ReLU.backward(ds, cache_relu)
        dx, dw, db = FastConv.backward(da, cache_conv)

        return dx, dw, db


# 2x2最大池化层
class ThreeLayerConvNet(object):
    def __init__(
        self,
        input_dims = (3, 32, 32),
        num_filters = 32,
        filter_size = 7,
        hidden_dim = 100,
        num_classes = 10,
        weight_scale = 1e-3,
        reg = 0.0,
        dtype = torch.float,
        device = 'cpu'
    ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dims

        # 卷积层权重 - 尺寸是 (N, C, H, W)
        self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters, C, filter_size, filter_size), dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)

        # 展平和全连接层 - 经过2x2最大池化后的图像尺寸变为原来的一半，同时将其展平
        self.params['W2'] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters * (H // 2) * (W // 2), hidden_dim), dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        # 输出层
        self.params['W3'] = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'param': self.params,
            'reg': self.reg,
            'dtype': self.dtype
        }
        torch.save(checkpoint, path)
        print("checkpoint saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['param']
        self.reg = checkpoint['reg']
        self.dtype = checkpoint['dtype']

        print("load checkpoint in {}".format(path))

    def loss(self, X, y = None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        """
            pad公式: Output_size = 1 + (Input_size - filter_size + 2 * padding) // stride
            这里希望在卷积后输出特征图与输入图像的尺寸相同 则: Output_size = Input_size
            又 stride = 1
        """
        conv_param = {
            'stride': 1,
            'pad': (filter_size - 1) // 2
        }
        pool_param = {
            'stride': 2,
            'pool_height': 2,
            'pool_width': 2
        }

        scores = None

        out_1, cache_1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out_2, cache_2 = Linear_ReLU.forward(out_1, W2, b2)
        scores, cache_3 = Linear.forward(out_2, W3, b3)

        if y is None:
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2) + torch.sum(W3 * W3))

        dh3, grads['W3'], grads['b3'] = Linear.backward(dscores, cache_3)
        dh2, grads['W2'], grads['b2'] = Linear_ReLU.backward(dh3, cache_2)
        dh1, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dh2, cache_1)

        grads['W3'] += 2 * self.reg * self.params['W3']
        grads['W2'] += 2 * self.reg * self.params['W2']
        grads['W1'] += 2 * self.reg * self.params['W1']

        return loss, grads


class DeepConvNet(object):
    def __init__(
        self,
        input_dims = (3, 32, 32),
        num_filters = [8, 8, 8, 8, 8],
        max_pools = [0, 1, 2, 3, 4],
        num_classes = 10,
        batchnorm = False,
        reg = 0.0,
        weight_scale = 1e-3,
        weight_initializer = None,
        dtype = torch.float,
        device = 'cpu'
    ):
        """
                   所有卷积层使用 kernel_size = 3, pad = 1,  池化层使用 2x2, stride = 2
                   对于 num_filters = [8, 8, 8]，网络结构如下：
                   第1层：卷积层 + [批归一化] + ReLU + [池化层]（取决于 max_pools 设置）
                   第2层：卷积层 + [批归一化] + ReLU + [池化层]（取决于 max_pools 设置）
                   第3层：卷积层 + [批归一化] + ReLU + [池化层]（取决于 max_pools 设置）
                   第4层：全连接层（输出层）
                   max_pools列表表示使用池化层的索引, 这里默认全使用
        """
        self.params = {}
        L = self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        # 卷积层的权重 W{i} 的形状为 (out_channels, in_channels, kernel_size, kernel_size)
        # 卷积层的偏置 b{i} 的形状为 (out_channels,)
        C, H, W = input_dims
        if weight_scale == 'kaiming':
            self.params['W1'] = kaiming_initializer(C, num_filters[0], K=3, relu=True, dtype=dtype, device=device)
        else:
            self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters[0], C, 3, 3),
                                             dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters[0], dtype=dtype, device=device)
        if self.batchnorm:
            self.params['gamma1'] = torch.ones(num_filters[0], dtype=dtype, device=device)
            self.params['beta1'] = torch.zeros(num_filters[0], dtype=dtype, device=device)


        for i in range(2, L):
            if weight_scale == 'kaiming':
                self.params[f'W{i}'] = kaiming_initializer(num_filters[i-2], num_filters[i-1], K=3, relu=True, dtype=dtype, device=device)
            else:
                self.params[f'W{i}'] = torch.normal(mean=0.0, std=weight_scale,
                                                    size=(num_filters[i - 1], num_filters[i - 2], 3, 3),
                                                    dtype=dtype, device=device)

            self.params[f'b{i}'] = torch.zeros(num_filters[i-1], dtype=dtype, device=device)
            if self.batchnorm:
                self.params[f'gamma{i}'] = torch.ones(num_filters[i - 1], dtype=dtype, device=device)
                self.params[f'beta{i}'] = torch.zeros(num_filters[i - 1], dtype=dtype, device=device)

        # 每次池化操作后图像宽高减半 , 共有len(max_pools)次池化操作
        H_out = H // (2 ** len(max_pools))
        W_out = W // (2 ** len(max_pools))
        dim_out = num_filters[-1] * H_out * W_out
        if weight_scale == 'kaiming':
            self.params[f'W{L}'] = kaiming_initializer(dim_out, num_classes, K=None, relu=False, dtype=dtype, device=device)
        else:
            self.params[f'W{L}'] = torch.normal(mean=0.0, std=weight_scale, size=(dim_out, num_classes),
                                                dtype=dtype, device=device)
        self.params[f'b{L}'] = torch.zeros(num_classes, dtype=dtype, device=device)


        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batchnorm,
            'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y = None):
        X = X.to(self.dtype)

        mode = 'test' if y is None else 'train'

        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        filter_size = 3
        conv_param = {
            'stride': 1,
            'pad': (filter_size - 1) // 2
        }
        pool_param = {
            'stride': 2,
            'pool_height': 2,
            'pool_width': 2
        }
        L = self.num_layers
        caches = []

        for i in range(L-1):
            w, b = self.params[f'W{i+1}'], self.params[f'b{i+1}']
            if self.batchnorm:
                gamma, beta = self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']
                bn_param = self.bn_params[i]
                if i in self.max_pools:
                    X, cache = Conv_BatchNorm_ReLU_Pool.forward(X, w, b, gamma, beta, conv_param, bn_param, pool_param)
                else:
                    X, cache = Conv_BatchNorm_ReLU.forward(X, w, b, gamma, beta, conv_param, bn_param)
            else:
                if i in self.max_pools:
                    X, cache = Conv_ReLU_Pool.forward(X, w, b, conv_param, pool_param)
                else:
                    X, cache = Conv_ReLU.forward(X, w, b, conv_param)
            caches.append(cache)

        scores, cache_fin = Linear.forward(X, self.params[f'W{L}'], self.params[f'b{L}'])
        caches.append(cache_fin)

        if y is None:
            return scores

        loss, grads = 0.0, {}
        loss, d_scores = softmax_loss(scores, y)
        for i in range(L):
            loss += self.reg * torch.sum(self.params[f'W{i + 1}'] ** 2)

        dh, grads[f'W{L}'], grads[f'b{L}'] = Linear.backward(d_scores, caches[-1])
        grads[f'W{L}'] += 2 * self.reg * self.params[f'W{L}']

        for i in range(L-1, 0, -1):
            if self.batchnorm:
                if i - 1 in self.max_pools:
                    dh, grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = Conv_BatchNorm_ReLU_Pool.backward(dh, caches[i-1])
                else:
                    dh, grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = Conv_BatchNorm_ReLU.backward(dh, caches[i-1])
            else:
                if i - 1 in self.max_pools:
                    dh, grads[f'W{i}'], grads[f'b{i}'] = Conv_ReLU_Pool.backward(dh, caches[i-1])
                else:
                    dh, grads[f'W{i}'], grads[f'b{i}'] = Conv_ReLU.backward(dh, caches[i-1])

            grads[f'W{i}'] += 2 * self.reg * self.params[f'W{i}']
        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-1
    learning_rate = 5e-4

    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None

    model = DeepConvNet(num_filters=[32, 64, 128],
                      max_pools=[1, 2],
                      reg=1e-5,
                      batchnorm=False,
                      # batchnorm=True,
                      weight_scale='kaiming',
                      device=device)
    solver = Solver(model, data_dict,
                    optim_config={'learning_rate': 2e-1},
                    lr_decay=0.98,
                    num_epochs=60, batch_size=128,
                    device=device, print_every=50)

    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu', dtype=torch.float32):
    """

    Args:
        Din:    输入维度 (神经元或通道数)
        Dout:   输出维度 (神经元或通道数)
        K:      卷积核大小，如果为 None 则初始化线性层的权重
        relu:   如果为 True 则使用增益值为 2 (用于ReLU激活)，否则为 1 使用 Xavier 初始化
        device:
        dtype:

    Returns:
        weight: 初始化后的权重张量，线性层的形状为 (Din, Dout), 卷积层的形状为 (Dout, Din, K, K)
    """
    # gain = 2 - ReLU激活     gain = 1 - Xavier初始化
    if relu:
        gain = 2
    else:
        gain = 1

    weight = None
    # 线性层
    if K is None:
        weight = torch.normal(0.0, (gain / Din) ** 0.5, size=(Din, Dout), dtype=dtype, device=device)
    # K不为None，则为卷积层
    else:
        weight = torch.normal(0.0, (gain / Din / K / K) ** 0.5, size=(Dout, Din, K, K), dtype=dtype, device=device)

    return weight


class BatchNorm(object):
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        BatchNorm
                在训练过程中: 对当前小批量数据计算均值与方差, 进行标准化,同时通过 momentum 控制累积均值与方差
        Args:
            x:
            gamma: 对标准化后的数据进行缩放
            beta: 对标准化后的数据进行平移
            bn_param:
                mode: 'train' or 'test'
                eps: prevent zero error
                momentum: 控制均值和方差更新速度的超参数
                running_mean & running_var: 在训练过程中保存累积的均值和方差, 在测试时使用保证数据的标准化和训练数据一致
        Returns: out, cache
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

        if mode == 'train':
            # 求对应特征维度的均值
            mean = torch.mean(x, dim=0)
            # unbiased = False (correction=0) 使用有偏方差 即除以N
            var = torch.var(x, dim=0, correction=0)
            sigma = torch.sqrt(var + eps)
            # 减均值除方差
            x_std = (x - mean) / sigma
            # 平移缩放
            out = gamma * x_std + beta
            cache = (mode, x, eps, gamma, mean, var, sigma, x_std)

            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var

        elif mode == 'test':
            sigma = torch.sqrt(running_var + eps)
            x_std = (x - running_mean) / sigma
            out = gamma * x_std + beta
            cache = (mode, x, eps, gamma, running_mean, running_var, sigma, x_std)

        else:
            raise ValueError('Invalid forward batchnorm mode %s' % mode)

        # 把更新后的running_mean和running_var存回bn_param字典中, 同时将其从计算图中分离出来以防止梯度传播提高效率
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        mode, x, eps, gamma, mean, var, sigma, x_std = cache
        N, D = x.shape

        if mode == 'train':

            dgamma = torch.sum(dout * x_std, dim=0)
            dbeta = torch.sum(dout, dim=0)

            dx_std = dout * gamma
            dvar = torch.sum(dx_std * (x - mean) * -0.5 * (var + eps) ** (-1.5), dim=0)
            dmean = torch.sum(dx_std * -1.0 / sigma, dim=0) + dvar * torch.sum(-2.0 * (x - mean) / N, dim=0)

            dx = dx_std * 1.0 / sigma + dvar * 2.0 * (x - mean) / N + dmean / N

        elif mode == 'test':
            dgamma = torch.sum(x_std * dout, dim=0)
            dbeta = torch.sum(dout, dim=0)
            # x^hat_i = (x_i - mu) / sigma 链式求导
            dx = dout * gamma / sigma
        else:
            raise ValueError('Invalid backward batchnorm mode "%s"' % mode)

        return dx, dgamma, dbeta


"""
    SpatialBatchNorm    对形状为 (N ,C, H, W)的x输入数据进行处理为形状 (N * H * W, C)
    批量归一化的核心思想是对每个特征维度进行标准化。在空间批量归一化中，我们希望对每个通道进行独立的标准化处理。因此，
    对于每个通道 C，我们需要汇总该通道在所有样本 N 和空间位置 H × W 上的统计信息(均值和方差)
    如果我们将输入数据重塑为形状 (N, C×H×W)，则每个通道的数据将不再独立，违背了批量归一化的设计
"""
class SpatialBatchNorm(object):
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        N, C, H, W = x.shape
        # reshape to (N * H * W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
        out_reshaped, cache = BatchNorm.forward(x_reshaped, gamma, beta, bn_param)
        # back to (N, C, H, W)
        out = out_reshaped.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        N, C, H, W = dout.shape
        dout_reshaped = dout.permute(0, 2, 3, 1).reshape(-1, C)
        temp, d_gamma, d_beta = BatchNorm.backward(dout_reshaped, cache)
        dx = temp.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return dx, d_gamma, d_beta


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta