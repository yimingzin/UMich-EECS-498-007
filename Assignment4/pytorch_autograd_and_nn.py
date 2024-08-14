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