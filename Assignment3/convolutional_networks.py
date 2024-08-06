import time

import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear, Linear_ReLU, Solver, adam, ReLU


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
            input_dims=(3, 32, 32),
            num_filters=[8, 8, 8, 8, 8],
            max_pools=[0, 1, 2, 3, 4],
            batchnorm=False,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
            weight_initializer=None,
            dtype=torch.float,
            device='cpu'
    ):
        """
            所有卷积层使用 kernel_size = 3, pad = 1,  池化层使用 2x2, stride = 2
            对于 num_filters = [8, 8, 8]，网络结构如下：
            第1层：卷积层 + [批归一化] + ReLU + [池化层]（取决于 max_pools 设置）
            第2层：卷积层 + [批归一化] + ReLU + [池化层]（取决于 max_pools 设置）
            第3层：卷积层 + [批归一化] + ReLU + [池化层]（取决于 max_pools 设置）
            第4层：全连接层（输出层）
            max_pools列表表示使用池化层的索引, 这里全使用
        """
        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        # 卷积层的权重 W{i} 的形状为 (out_channels, in_channels, kernel_size, kernel_size)
        # 卷积层的偏置 b{i} 的形状为 (out_channels,)
        C, H, W = input_dims
        self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters[0], C, 3, 3), dtype=dtype,
                                         device=device)
        self.params['b1'] = torch.zeros(num_filters[0], dtype=dtype, device=device)

        for i in range(2, self.num_layers):
            self.params[f'W{i}'] = torch.normal(mean=0.0, std=weight_scale,
                                                size=(num_filters[i - 1], num_filters[i - 2], 3, 3), dtype=dtype,
                                                device=device)
            self.params[f'b{i}'] = torch.zeros(num_filters[i - 1], dtype=dtype, device=device)

        # 每次池化操作后图像宽高减半 , 共有len(max_pools)次池化操作
        H_out = H // (2 ** len(max_pools))
        W_out = W // (2 ** len(max_pools))
        dim_out = num_filters[-1] * H_out * W_out
        self.params[f'W{self.num_layers}'] = torch.normal(mean=0.0, std=weight_scale, size=(dim_out, num_classes),
                                                          dtype=dtype, device=device)
        self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=dtype, device=device)

        # ----------------------------------------------------------------------------------------------
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

    def loss(self, X, y=None):
        X = X.to(self.dtype)

        filter_size = 3
        conv_param = {
            'stride': 1,
            'pad': (filter_size - 1) // 2
        }
        pool_param = {
            'pool_height': 2,
            'pool_width': 2,
            'stride': 2
        }
        scores = None
        caches = []
        for i in range(self.num_layers - 1):
            w, b = self.params[f'W{i + 1}'], self.params[f'b{i + 1}']
            if i in self.max_pools:
                X, cache_i = Conv_ReLU_Pool.forward(X, w, b, conv_param, pool_param)
            else:
                X, cache_i = Conv_ReLU.forward(X, w, b, conv_param)
            caches.append(cache_i)

        w, b = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, cache_fin = Linear.forward(X, w, b)
        caches.append(cache_fin)

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, d_scores = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += self.reg * torch.sum(self.params[f'W{i + 1}'] ** 2)

        dx, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(d_scores, caches[-1])
        grads[f'W{self.num_layers}'] += 2 * self.reg * self.params[f'W{self.num_layers}']

        for i in range(self.num_layers - 1, 0, -1):
            if i - 1 in self.max_pools:
                dx, grads[f'W{i}'], grads[f'b{i}'] = Conv_ReLU_Pool.backward(dx, caches[i - 1])
            else:
                dx, grads[f'W{i}'], grads[f'b{i}'] = Conv_ReLU.backward(dx, caches[i - 1])

            grads[f'W{i}'] += 2 * self.reg * self.params[f'W{i}']

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-1
    learning_rate = 5e-4

    return weight_scale, learning_rate
