import math
from typing import Optional, Tuple
import torch
import torchvision
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torchvision.models import feature_extraction


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for a
    tiny RegNet model so it can train decently with a single K80 Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


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

    def forward(self, x):
        out = self.W_embed[x]
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    N, T, V = x.shape
    N, T = y.shape

    x_flatten = x.reshape(N * T, V)
    y_flatten = y.reshape(N * T)

    loss = F.cross_entropy(x_flatten, y_flatten, ignore_index=ignore_index, reduction='sum')
    loss /= N

    return loss


class CaptioningRNN(nn.Module):
    def __init__(self, word_to_idx, input_dim: int = 512, wordvec_dim=128,
                 hidden_dim: int = 128, cell_type: str = 'rnn',
                 image_encoder_pretrained: bool = True, ignore_index: Optional[int] = None):
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'attn'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index

        self.image_encoder = ImageEncoder()
        if self.cell_type == 'rnn':
            self.project_input = nn.Linear(input_dim * 4 * 4, hidden_dim)
        # add LSTM
        elif self.cell_type == 'lstm':
            self.project_input = nn.Linear(input_dim * 4 * 4, hidden_dim)
        # add LSTM Attention
        elif self.cell_type == 'attn':
            self.project_input = nn.Linear(input_dim, hidden_dim)

        self.word_embed = WordEmbedding(vocab_size, wordvec_dim)

        if self.cell_type == 'rnn':
            self.network = RNN(wordvec_dim, hidden_dim)
        # add LSTM
        elif self.cell_type == 'lstm':
            self.network = LSTM(wordvec_dim, hidden_dim)
        # add LSTM Attention
        elif self.cell_type == 'attn':
            self.network = AttentionLSTM(wordvec_dim, hidden_dim)

        self.project_output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        image_encoder = self.image_encoder.forward(images)
        image_encoder_attn = image_encoder.permute(0, 2, 3, 1)

        if self.cell_type == 'rnn':
            h = self.project_input.forward(image_encoder.reshape(image_encoder.shape[0], -1))
        elif self.cell_type == 'lstm':
            h = self.project_input.forward(image_encoder.reshape(image_encoder.shape[0], -1))
        elif self.cell_type == 'attn':
            A = self.project_input.forward(image_encoder_attn).permute(0, 3, 1, 2)


        x = self.word_embed.forward(captions_in)

        if self.cell_type == 'rnn':
            h = self.network.forward(x, h)
        elif self.cell_type == 'lstm':
            h = self.network.forward(x, h)
        elif self.cell_type == 'attn':
            h = self.network.forward(x, A)

        scores = self.project_output.forward(h)

        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self.ignore_index)

        return loss

    def sample(self, images, max_length=15):

        N = images.shape[0]

        captions = images.new(N, max_length).fill_(1).long() * self._null
        words = images.new(N, 1).fill_(1).long() * self._start
        if self.cell_type == 'attn':
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(1).float()

        image_encoder = self.image_encoder.forward(images)
        image_encoder_attn = image_encoder.permute(0, 2, 3, 1)

        if self.cell_type == 'rnn':
            h = self.project_input.forward(image_encoder.reshape(image_encoder.shape[0], -1))
        elif self.cell_type == 'lstm':
            h = self.project_input.forward(image_encoder.reshape(image_encoder.shape[0], -1))
            c = torch.zeros_like(h)
        elif self.cell_type == 'attn':
            A = self.project_input.forward(image_encoder_attn).permute(0, 3, 1, 2)
            h = A.mean(dim=(2, 3))
            c = A.mean(dim=(2, 3))

        for i in range(max_length):
            x = self.word_embed.forward(words).reshape(N, -1)
            if self.cell_type == 'rnn':
                h = self.network.step_forward(x, h)
            elif self.cell_type == 'lstm':
                h, c = self.network.step_forward(x, h, c)
            elif self.cell_type == 'attn':
                attn, attn_weights_all[:, i, :, :] = dot_product_attention(h, A)
                h, c = self.network.step_forward(x, h, c, attn)

            scores = self.project_output.forward(h)
            words = torch.argmax(scores, dim=1)
            captions[:, i] = words

        if self.cell_type == 'attn':
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', dtype=torch.float32):
        super().__init__()
        self.Wx = Parameter(
            torch.randn((input_size, hidden_size * 4), dtype=dtype, device=device).div(math.sqrt(input_size)))
        self.Wh = Parameter(
            torch.randn((hidden_size, hidden_size * 4), dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size * 4, dtype=dtype, device=device))

    def step_forward(self, x: torch.Tensor, prev_h: torch.Tensor,
                     prev_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, H = prev_h.shape
        a = torch.mm(x, self.Wx) + torch.mm(prev_h, self.Wh) + self.b

        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2 * H])
        o = torch.sigmoid(a[:, 2 * H:3 * H])
        g = torch.tanh(a[:, 3 * H:])

        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)

        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        N, H = h0.shape

        c0 = torch.zeros_like(h0)

        h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)

        prev_h = h0
        prev_c = c0

        for t in range(T):
            next_h, next_c = self.step_forward(x[:, t, :], prev_h, prev_c)
            h[:, t, :] = next_h
            prev_h = next_h
            prev_c = next_c

        return h


def dot_product_attention(prev_h, A):

    N, H, D_a, _ = A.shape
    A_flatten = A.reshape(N, H, -1)
    prev_h = prev_h.reshape(N, H, 1).permute(0, 2, 1)

    attn_scores = torch.bmm(prev_h, A_flatten) / (H ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=2)
    attn = torch.bmm(A_flatten, attn_weights.permute(0, 2, 1))

    attn = attn.reshape(N, H)
    attn_weights = attn_weights.reshape(N, D_a, D_a)

    return attn, attn_weights

class AttentionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.Wx = Parameter(torch.randn((input_size, hidden_size * 4)).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn((hidden_size, hidden_size * 4)).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size * 4))
        self.Wattn = Parameter(torch.randn((hidden_size, hidden_size * 4)).div(math.sqrt(hidden_size)))

    def step_forward(self, x: torch.Tensor, prev_h: torch.Tensor,
                     prev_c: torch.Tensor, attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        N, H = prev_h.shape
        a = torch.mm(x, self.Wx) + torch.mm(prev_h, self.Wh) + torch.mm(attn, self.Wattn) + self.b

        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])

        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)

        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):

        N, T, D = x.shape
        H = A.shape[1]

        h0 = torch.mean(A, dim=(2, 3))
        c0 = h0

        h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
        prev_h = h0
        prev_c = c0

        for t in range(T):
            attn, attn_weights = dot_product_attention(prev_h, A)
            next_h, next_c = self.step_forward(x[:, t, :], prev_h, prev_c, attn)
            h[:, t, :] = next_h
            prev_h = next_h
            prev_c = next_c

        return h

