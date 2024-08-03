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

        # 计算输出数据的高度H' 和宽度W' (确保为整数)
        H_prime = 1 + (H - HH + 2 * pad) // stride
        W_prime = 1 + (W - WW + 2 * pad) // stride

        # 初始化输出数据
        out = torch.zeros((N, C, H_prime, W_prime), dtype=x.dtype, device=x.device)

        # 对输入数据进行0填充保证输出大小相同
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # 当前处理图片的索引
        for n in range(N):
            # 滤波器索引
            for f in range(F):
                # 下面两层循环是为了计算卷积核(感受野)在输入图像的起始和结束位置
                for h_prime in range(H_prime):
                    for w_prime in range(W_prime):
                        h_start, h_end = h_prime * stride, h_prime * stride + HH
                        w_start, w_end = w_prime * stride, w_prime * stride + WW

                        # 从填充后的图像提取当前感受野
                        receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]
                        # 计算卷积操作的输出, 将当前感受野与滤波器w[f]对应元素相乘求和加上偏置
                        out[n, f, h_prime, w_prime] = torch.sum(receptive_field * w[f]) + b[f]
        cache = (x, w, b, conv_param)
        return out, cache
