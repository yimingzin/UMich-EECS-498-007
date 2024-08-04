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

                        mask = (receptive_field == torch.max(receptive_field))
                        dx[n, c, h_start:h_end, w_start:w_end][mask] += dout[n, c, h_prime, w_prime]
        return dx
