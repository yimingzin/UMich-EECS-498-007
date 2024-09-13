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
    prev_h = h0
    cache = []

    for t in range(T):
        next_h, cache_step = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        prev_h = next_h
        cache.append(cache_step)

    return h, cache

def rnn_backward(dh, cache):
    N, T, H = dh.shape
    N, D = cache[0][0].shape

    dx = torch.zeros((N, T, D), dtype=dh.dtype, device=dh.device)
    dprev_h = torch.zeros_like(cache[0][1], dtype=dh.dtype, device=dh.device)
    dWx = torch.zeros_like(cache[0][2], dtype=dh.dtype, device=dh.device)
    dWh = torch.zeros_like(cache[0][3], dtype=dh.dtype, device=dh.device)
    db = torch.zeros_like(cache[0][4], dtype=dh.dtype, device=dh.device)

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
        self.Wx = Parameter(torch.randn((input_size, hidden_size), dtype=dtype, device=device).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn((hidden_size, hidden_size), dtype=dtype, device=device).div(math.sqrt(hidden_size)))
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
        self.W_embed = Parameter(torch.randn((vocab_size, embed_size), dtype=dtype, device=device).div(math.sqrt(vocab_size)))

    #把x从离散的单词索引转换为连续的向量表示
    # x.shape = (N, T) N是句子的数量，T是每个句子的单词数量
    '''
    x = [
        [4, 12, 5, 407, 0],  # 第一个句子：'a man on a bicycle <NULL>'
        [7, 20, 15, 12, 8]   # 第二个句子：'the next two men in'
    ]
    '''
    def forward(self, x):
        return self.W_embed[x]

def temporal_softmax_loss(x, y, ignore_index = None):
    N, T, V = x.shape
    N, T = y.shape

    x_flat = x.reshape(N*T, V)
    y_flat = y.reshape(N*T)

    loss = F.cross_entropy(x_flat, y_flat, ignore_index=ignore_index, reduction='sum')
    loss /= N

    return loss

class CaptioningRNN(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', device='cpu',
                 ignore_index = None, dtype=torch.float32):
        '''

        :param word_to_idx: 数据集: 字典 {word : index}
        :param input_dim: 如果为 rnn / lstm = 1280 else = 1280 * 4 * 4
        :param wordvec_dim: 词嵌入矩阵维度
        :param hidden_dim:
        :param cell_type: rnn or lstm
        '''
        super().__init__()

        if cell_type not in {'rnn', 'lstm', 'attention'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        # 生成一个索引对应单词的字典
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        # 取出 <NULL> <START> <END> 对应的索引
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index

        if self.cell_type in ['rnn', 'lstm']:
            self.feature_extractor = FeatureExtractor(pooling=True, device=device, dtype=dtype)
        else:
            self.feature_extractor = FeatureExtractor(pooling=False, device=device, dtype=dtype)

        self.project_feature = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        self.word_embedding = WordEmbedding(vocab_size, wordvec_dim, device=device, dtype=dtype)

        if self.cell_type == 'rnn':
            self.network = RNN(wordvec_dim, hidden_dim, device=device, dtype=dtype)

        self.project_output = nn.Linear(hidden_dim, vocab_size).to(device=device, dtype=dtype)

    def forward(self, images, captions):
        '''
        :param images: Input images, of shape (N, 3, 112, 112)
        :param captions:  (N, T)  N是批次大小表示有多少句子， T是每个句子长度
        :return:
        '''

        # captions_in 和 captions_out 的形状都是 (N, T-1)
        # captions_in 告诉模型当前处理的单词是什么
        # captions_out 告诉模型下一个正确的单词是什么
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0

        word_vectors = self.word_embedding.forward(captions_in)
        image_features = self.feature_extractor.extract_mobilenet_feature(images)
        if self.cell_type in ['rnn', 'lstm']:
            h0 = self.project_feature(image_features)
            hT = self.network.forward(word_vectors, h0)

        scores = self.project_output(hT)
        loss = temporal_softmax_loss(scores, captions_out, self.ignore_index)

        return loss

    def sample(self, images, max_length=15):
        N = images.shape[0]
        # images.new() 创建了形状为(N, max_length) device和dtype都和images相同的张量，
        # 填充为1转为长整型乘以self._null -> <NULL>对应的索引全部初始化为<NULL>
        captions = self._null * images.new(N, max_length).fill_(1).long()

        image_features = self.feature_extract.extract_mobilenet_feature(images)
        if self.cell_type == 'rnn':
            h = self.project_input(image_features)

        # 初始化一个 [N, 1] 值全部为<START>索引的张量， 确保生成的描述从这个单词开始
        words = self._start * images.new(N, 1).long()

        for i in range(max_length):
            x = self.word_embed.forward(words).reshape(N, -1)
            if self.cell_type == 'rnn':
                h = self.network.step_forward(x, h)

            scores = self.project_output(h)
            words = torch.argmax(scores, dim=1)
            # 在每个单个时间步里模型同时处理N张图片，对N张图片生成一个单词，所以用列表示单词
            captions[:, i] = words

        return captions
