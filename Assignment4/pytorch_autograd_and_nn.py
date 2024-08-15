import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

from eecs598 import reset_seed

to_float = torch.float
to_long = torch.long
loader_train, loader_val, loader_test = load_CIFAR(path='./datasets/')


def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim, end_dim)


def check_accuracy_part2(loader, model_fn, params):
    split = 'val' if loader.dataset.train else 'test'
    print('Now check accuracy on %s set ' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dtype=to_float, device='cuda')
            y = y.to(dtype=to_long, device='cuda')
            scores = model_fn(x, params)
            preds = torch.argmax(scores, dim=1)
            num_correct += (y == preds).sum()
            num_samples += preds.size(dim=0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct, accuracy = %.2f%%' % (num_correct, num_correct, 100 * acc))
    return acc


def train_part2(model_fn, params, learning_rate):
    for t, (x, y) in enumerate(loader_train):
        x = x.to(dtype=to_float, device='cuda')
        y = y.to(dtype=to_long, device='cuda')
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)
        loss.backward()

        with torch.no_grad():
            for w in params:
                if w.requires_grad:
                    w -= learning_rate * w.grad
                    w.grad.zero_()

        if t % 100 == 0 or t == len(loader_train) - 1:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            acc = check_accuracy_part2(loader_val, model_fn, params)

    return acc


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Now checking accuracy on val set')
    else:
        print('Now checking accuracy on test set')
    num_correct, num_samples = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dtype=to_float, device='cuda')
            y = y.to(dtype=to_long, device='cuda')
            scores = model(x)
            preds = torch.argmax(scores, dim=1)
            num_correct += (y == preds).sum()
            num_samples += preds.size(dim=0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct, accuracy = %.2f%%' % (num_correct, num_samples, acc * 100))
    return acc


def adjust_learning_rate(optimizer, epoch, schedule, learning_rate_decay):
    """
    optimizer.param_groups = [
        {
            'params': [list of parameters],
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        },
        {
            'params': [another list of parameters],
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }
    ]
    """
    if epoch in schedule:
        for param_groups in optimizer.param_groups:
            print('learning_rate decay from {} to {}'.format(param_groups['lr'],
                                                             param_groups['lr'] * learning_rate_decay))
            param_groups['lr'] *= learning_rate_decay


def train_part345(optimizer, model, epoch=1, schedule=[], learning_rate_decay=.1, verbose=True):
    model = model.to(device='cuda')
    print_every = 100
    num_iters = epoch * len(loader_train)
    if verbose:
        num_prints = num_iters // print_every + 1
    else:
        num_prints = epoch

    acc_history = torch.zeros(num_prints, dtype=to_float)
    iter_history = torch.zeros(num_prints, dtype=to_long)

    for e in range(epoch):
        adjust_learning_rate(optimizer, e, schedule, learning_rate_decay)

        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(dtype=to_float, device='cuda')
            y = y.to(dtype=to_long, device='cuda')
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tt = t + e * len(loader_train)
            if verbose and (tt % print_every == 0 or (e == epoch - 1 and t == len(loader_train) - 1)):
                print('epoch: %d, Iteration: %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[tt // print_every] = acc
                iter_history[tt // print_every] = tt
                print()
            elif not verbose and (t == len(loader_train) - 1):
                print('epoch: %d, Iteration: %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[e] = acc
                iter_history[e] = tt
                print()
    return acc_history, iter_history


def two_layer_fc(x, params):
    w1, b1, w2, b2 = params
    x = flatten(x)
    x = F.relu(F.linear(x, w1, b1))
    scores = F.linear(x, w2, b2)

    return scores


def two_layer_fc_test():
    hidden_layer_size = 4200
    num_classes = 10
    x = torch.zeros((64, 3, 32, 32), dtype=to_float)
    w1 = torch.zeros((hidden_layer_size, 3 * 32 * 32), dtype=to_float)
    b1 = torch.zeros((hidden_layer_size,), dtype=to_float)
    w2 = torch.zeros((num_classes, hidden_layer_size), dtype=to_float)
    b2 = torch.zeros((num_classes,), dtype=to_float)

    params = [w1, b1, w2, b2]

    scores = two_layer_fc(x, params)
    print('scores shape:', list(scores.size()))


def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    x = F.relu(F.conv2d(x, conv_w1, conv_b1, stride=1, padding=2))
    x = F.relu(F.conv2d(x, conv_w2, conv_b2, stride=1, padding=1))
    x = flatten(x)
    scores = F.linear(x, fc_w, fc_b)

    return scores


def three_layer_convnet_test():
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    kernel_size_2 = 3

    x = torch.zeros((64, C, H, W), dtype=to_float)
    conv_w1 = torch.zeros((channel_1, C, kernel_size_1, kernel_size_1), dtype=to_float)
    conv_b1 = torch.zeros((channel_1,), dtype=to_float)
    conv_w2 = torch.zeros((channel_2, channel_1, kernel_size_2, kernel_size_2), dtype=to_float)
    conv_b2 = torch.zeros((channel_2,), dtype=to_float)
    fc_w = torch.zeros((num_classes, channel_2 * H * W), dtype=to_float)
    fc_b = torch.zeros((num_classes,), dtype=to_float)

    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    scores = three_layer_convnet(x, params)
    print('scores shape: ', list(scores.size()))


def initialize_two_layer_fc():
    C, H, W = 3, 32, 32
    num_classes = 10

    hidden_size = 4200

    w1 = nn.init.kaiming_normal_(torch.empty(hidden_size, C * H * W, dtype=to_float, device='cuda'))
    b1 = nn.init.zeros_(torch.empty(hidden_size, dtype=to_float, device='cuda'))
    w2 = nn.init.kaiming_normal_(torch.empty(num_classes, hidden_size, dtype=to_float, device='cuda'))
    b2 = nn.init.zeros_(torch.empty(num_classes, dtype=to_float, device='cuda'))
    w1.requires_grad = True
    b1.requires_grad = True
    w2.requires_grad = True
    b2.requires_grad = True

    params = [w1, b1, w2, b2]
    return params


def initializer_three_layer_conv_part2():
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    kernel_size_2 = 3

    conv_w1 = nn.init.kaiming_normal_(
        torch.empty(channel_1, C, kernel_size_1, kernel_size_1, dtype=to_float, device='cuda'))
    conv_b1 = nn.init.zeros_(torch.empty(channel_1, dtype=to_float, device='cuda'))
    conv_w2 = nn.init.kaiming_normal_(
        torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, dtype=to_float, device='cuda'))
    conv_b2 = nn.init.zeros_(torch.empty(channel_2, dtype=to_float, device='cuda'))

    fc_w = nn.init.kaiming_normal_(torch.empty(num_classes, channel_2 * H * W, dtype=to_float, device='cuda'))
    fc_b = nn.init.zeros_(torch.empty(num_classes, dtype=to_float, device='cuda'))

    conv_w1.requires_grad = True
    conv_b1.requires_grad = True
    conv_w2.requires_grad = True
    conv_b2.requires_grad = True
    fc_w.requires_grad = True
    fc_b.requires_grad = True

    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]

    return params


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = flatten(x)
        scores = F.relu(self.fc1(x))
        scores = self.fc2(scores)
        return scores


def test_TwoLayerFC():
    x = torch.zeros((64, 3, 16, 16), dtype=to_float)
    model = TwoLayerFC(3 * 16 * 16, 4200, 10)
    scores = model(x)
    print('Architecture: ')
    print(model)
    print('Output size: ', list(scores.size()))


class ThreeLayerConvNet(nn.Module):
    def __init__(self, input_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(input_channel, channel_1, kernel_size=5, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)

        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.kaiming_normal_(self.conv_2.weight)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.conv_1.bias)
        nn.init.zeros_(self.conv_2.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = flatten(x)
        scores = self.fc(x)
        return scores


def test_ThreeLayerConvNet():
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 16
    channel_2 = 8

    x = torch.zeros((64, C, H, W), dtype=to_float)
    model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes)
    scores = model(x)
    print('Architecture: ')
    print(model)
    print('Output size: ', list(scores.size()))


def initialize_two_layer_fc_part3():
    reset_seed(0)
    C, H, W = 3, 32, 32
    num_classes = 10
    hidden_layer_size = 4000
    learning_rate = 1e-2
    weight_decay = 1e-4

    model = TwoLayerFC(C * H * W, hidden_layer_size, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _ = train_part345(optimizer, model)


def initializer_three_layer_conv_part3():
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 32
    channel_2 = 16

    learning_rate = 3e-3
    weight_decay = 1e-4

    model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, optimizer


def initialize_two_layer_part4():
    C, H, W = 3, 32, 32
    num_classes = 10

    hidden_layer_size = 4200
    learning_rate = 1e-2
    weight_decay = 1e-4
    momentum = 0.5

    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(C*H*W, hidden_layer_size)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer_size, num_classes))
    ]))

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    return model, optimizer

def initialize_three_layer_conv_part4():
    C, H, W = 3, 32, 32
    num_classes = 10

    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    pad_size_1 = 2

    kernel_size_2 = 3
    pad_size_2 = 1

    learning_rate = 1e-2
    weight_decay = 1e-4
    momentum = 0.5

    model = nn.Sequential(OrderedDict([
        ('conv_1', nn.Conv2d(C, channel_1, kernel_size_1, stride=1, padding=pad_size_1)),
        ('relu_1', nn.ReLU()),
        ('conv_2', nn.Conv2d(channel_1, channel_2, kernel_size_2, stride=1, padding=pad_size_2)),
        ('relu_2', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('fc', nn.Linear(channel_2*H*W, num_classes))
    ]))

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    return model, optimizer

##############################################################################################
# Part V. ResNet for CIFAR-10
##############################################################################################

class PlainBlock(nn.Module):
    def __init__(self, Cin, Cout, downsample = False):
        """
        Spatial Batch Normalization => ReLU => Conv_1 3x3 padding = 1 , if down_sample = True stride=2 else = 1
        => Spatial Batch Normalization => ReLU => Conv_2 3x3 padding = 1
        """
        super().__init__()
        self.net = None
        layer = []
        # 1. Spatial Batch normalization
        layer.append(nn.BatchNorm2d(Cin))
        # 2. ReLU, 使用inplace = True 会回改变输入张量
        layer.append(nn.ReLU(inplace=True))
        # 3. Convolutional layer with Cout 3x3 filters
        downsample_stride = 2 if downsample else 1
        # 4. Conv_1, 从Cin特征图的通道数到Cout的特征图通道数 (一般是从小到大)
        layer.append(nn.Conv2d(Cin, Cout, kernel_size=3, stride=downsample_stride, padding=1))
        # 5. Spatial Batch normalization, 对Cout输出的特征图通道数做批归一化
        layer.append(nn.BatchNorm2d(Cout))
        # 6. ReLU
        layer.append(nn.ReLU(inplace=True))
        # 6. Convolutional layer with Cout 3x3 filters, 当我们希望在当前的通道数上提取更多信息时，且保持不改变特征图的宽度和深度
        layer.append(nn.Conv2d(Cout, Cout, kernel_size=3, stride=1, padding=1))

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, Cin, Cout, downsample = False):
        """
            残差块的实现：假设 F 是 Plain块, 则残差块计算如下
            R(x) = F(x) + x         - 但是这个公式只有 x 与 F(x)的形状相同时才符合，实际情况有两种原因导致输入与输出不同：
            1. 输出通道数 Cout 不等于 输入通道数 Cin
            2. Plain 块进行了下采样
            所以对公式进行推广成为： R(x) = F(x) + G(x)
            对 G(x) 进行分类讨论
                1. 不进行下采样       - Cin = Cout 则 G(x) = x
                                    - Cin != Cout 则 G(x) 是一个1x1 stride = 1输出为 Cout 的卷积
                2. 进行下采样  - 执行相同的下采样操作  所以 G(x) 是一个 1x1 stride = 2 输出为 Cout 的卷积
        """
        super().__init__()
        self.block = None
        self.shortcut = None

        self.block = PlainBlock(Cin, Cout, downsample)

        if not downsample:
            if Cin == Cout:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(Cin, Cout, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Conv2d(Cin, Cout, kernel_size=1, stride=2)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class ResNetStage(nn.Module):
    def __init__(self, Cin, Cout, num_blocks, downsample=True, block=ResidualBlock):
        """
        :param num_blocks: block的数量
        :param block: 使用什么block
        """
        super().__init__()
        blocks = [block(Cin, Cout, downsample)]
        for _ in range(num_blocks - 1):
            blocks.append(block(Cout, Cout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class ResNetStem(nn.Module):
    """
        ResNet的 Stem 是网络的最开始部分， 通常用于对输入图像进行初步的特征提取
        Cin = 3 对应 RGB, Cout = 8 表示处理后特征图深度为 8
    """
    def __init__(self, Cin = 3, Cout = 8):
        super().__init__()
        layers = [
            nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, stride=1)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, stage_args, Cin=3, block=ResidualBlock, num_classes=10):
        """
             :param stage_args: 一个列表，每个元素是一个元组，描述了每个 ResNet 阶段的参数。
             每个元组包含四个元素：Cin（输入通道数）、Cout（输出通道数）、num_blocks（残差块数量）和 downsample（是否下采样）
        """
        super().__init__()
        self.cnn = None
        layers = [ResNetStem(Cin, stage_args[0][0])]
        for Cin, Cout, num_blocks, downsample in stage_args:
            layers.append(ResNetStage(Cin, Cout, num_blocks, downsample, block=block))
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(stage_args[-1][1], num_classes)

    def forward(self, x):
        scores = None
        x = self.cnn(x)
        N, C, H, W = x.shape
        # 生成一个 1x1的特征图
        # x = nn.AvgPool2d((H, W), stride=1)(x)
        avg_pool = nn.AvgPool2d((H, W), stride=1)
        x = avg_pool(x)
        x = flatten(x)
        scores = self.fc(x)

        return scores