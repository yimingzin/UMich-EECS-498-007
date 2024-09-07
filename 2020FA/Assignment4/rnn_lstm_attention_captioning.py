import torch
import math
import torch.nn as nn
from a4_helper import *
from torch.nn.parameter import Parameter
from torchsummary import summary


class FeatureExtractor(object):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, pooling=False, verbose=False,
                 device='cpu', dtype=torch.float32):

        from torchvision import transforms, models
        from torchsummary import summary
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device, self.dtype = device, dtype
        self.mobilenet = models.mobilenet_v2(pretrained=True).to(device)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])  # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(4, 4))  # input: N x 1280 x 4 x 4

        self.mobilenet.eval()
        if verbose:
            summary(self.mobilenet, (3, 112, 112))

    def extract_mobilenet_feature(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape N x 3 x 112 x 112

        Outputs:
        - feat: Image feature, of shape N x 1280 (pooled) or N x 1280 x 4 x 4
        """
        num_img = img.shape[0]

        img_prepro = []
        for i in range(num_img):
            img_prepro.append(self.preprocess(img[i].type(self.dtype).div(255.)))
        img_prepro = torch.stack(img_prepro).to(self.device)

        with torch.no_grad():
            feat = []
            process_batch = 500
            for b in range(math.ceil(num_img / process_batch)):
                feat.append(self.mobilenet(img_prepro[b * process_batch:(b + 1) * process_batch]
                                           ).squeeze(-1).squeeze(-1))  # forward and squeeze
            feat = torch.cat(feat)

            # add l2 normalization
            F.normalize(feat, p=2, dim=1)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h = torch.tanh(torch.mm(x, Wx) + torch.mm(prev_h, Wh) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    x, prev_h, Wx, Wh, b, next_h = cache
    dtanh = dnext_h * (1 - next_h ** 2)

    dx = torch.mm(dtanh, Wx.t())
    dprev_h = torch.mm(dtanh, Wh.t())
    dWx = torch.mm(x.t(), dtanh)
    dWh = torch.mm(prev_h.t(), dtanh)
    db = torch.sum(dtanh, dim=0)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    N, H = h0.shape

    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    cache = []
    prev_h = h0

    for t in range(T):
        next_h, cache_step = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(cache_step)
        prev_h = next_h

    return h, cache


def rnn_backward(dh, cache):
    N, T, H = dh.shape
    N, D = cache[0][0].shape

    dx = torch.zeros((N, T, D), dtype=cache[0][0].dtype, device=cache[0][0].device)
    dprev_h = torch.zeros_like(cache[0][1], dtype=cache[0][1].dtype, device=cache[0][2].device)
    dWx = torch.zeros_like(cache[0][2], dtype=cache[0][2].dtype, device=cache[0][2].device)
    dWh = torch.zeros_like(cache[0][3], dtype=cache[0][3].dtype, device=cache[0][3].device)
    db = torch.zeros_like(cache[0][4], dtype=cache[0][4].dtype, device=cache[0][4].device)

    for t in reversed(range(T)):
        dnext_h = dh[:, t, :] + dprev_h
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', dtype=torch.float32):
        super().__init__()
        self.Wx = Parameter(
            torch.randn((input_size, hidden_size), dtype=dtype, device=device).div(math.sqrt(input_size)))
        self.Wh = Parameter(
            torch.randn((hidden_size, hidden_size), dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

    def forward(self, x, h0):
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, device='cpu', dtype=torch.float32):
        super().__init__()
        self.W_embed = Parameter(
            torch.randn((vocab_size, embed_size), dtype=dtype, device=device).div(math.sqrt(vocab_size)))

    # x.shape = (N, T), 表示一个批量中有N个样本，每个样本包含T个单词，这里是17个
    # W_embed.shape = (vocab_size, embed_size), W_embed[x].shape = (N, T, embed_size)
    def forward(self, x):
        out = self.W_embed[x]
        return out
