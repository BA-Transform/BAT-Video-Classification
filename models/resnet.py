import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from collections import OrderedDict


try:
    from .non_local import NonLocalModule, get_nonlocal_block
except:
    from non_local import NonLocalModule, get_nonlocal_block


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, group=1, use_temp_conv=1, temp_stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes,
                               kernel_size=[1 + use_temp_conv * 2, 3, 3],
                               stride=[temp_stride, stride, stride],
                               padding=[use_temp_conv, 1, 1],
                               groups=group,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=[1, 3, 3],
                               stride=[1, 1, 1],
                               padding=[0, 1, 1],
                               groups=group,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, group=1, use_temp_conv=1, temp_stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes,
                               kernel_size=[1 + use_temp_conv * 2, 1, 1],
                               stride=[temp_stride, 1, 1],
                               padding=[use_temp_conv, 0, 0],
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=[1, 3, 3],
                               stride=[1, stride, stride],
                               padding=[0, 1, 1],
                               groups=group,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4,
                               kernel_size=1,
                               stride=[1, 1, 1],
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_temp_convs_set, temp_strides_set,
                 num_classes=400, dropout=0.5, group=1,
                 nonlocal_mod=1000, nltype='nl3d', inplanes=64,
                 k=4, tk=1, ts=4, nl_drop=0.2):
        self.inplanes = inplanes
        super(ResNet, self).__init__()
        self.nltype = nltype
        self.k = k
        self.tk = tk
        self.ts = ts
        self.nl_drop = nl_drop
        if type(nonlocal_mod) is int:
            nonlocal_mod = [nonlocal_mod] * 2
        while len(nonlocal_mod) < 2:
            nonlocal_mod.append(nonlocal_mod[-1])
        self.conv1 = nn.Conv3d(3, inplanes,
                               kernel_size=[
                                   1 + use_temp_convs_set[0][0] * 2, 7, 7],
                               stride=[temp_strides_set[0][0], 2, 2],
                               padding=[use_temp_convs_set[0][0], 3, 3],
                               bias=False)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)
        self.layer1 = self._make_layer(block, inplanes, layers[0], group=group, use_temp_convs=use_temp_convs_set[
                                       1], temp_strides=temp_strides_set[1])
        self.temp_pool = nn.MaxPool3d(kernel_size=[2, 1, 1], stride=[2, 1, 1])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, group=group, use_temp_convs=use_temp_convs_set[
                                       2], temp_strides=temp_strides_set[2], nonlocal_mod=nonlocal_mod[0])
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, group=group, use_temp_convs=use_temp_convs_set[
                                       3], temp_strides=temp_strides_set[3], nonlocal_mod=nonlocal_mod[1])
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2, group=group, use_temp_convs=use_temp_convs_set[
                                       4], temp_strides=temp_strides_set[4])
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Conv3d(512 * block.expansion, num_classes,
                            kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                if len(list(m.modules())) <= 1 and len(list(m.parameters())) > 0:
                    raise Exception("Unknown module:", type(m))
        for m in self.modules():
            if isinstance(m, NonLocalModule):
                m.init_modules()

    def _make_layer(self, block, planes, blocks, stride=1, group=1, use_temp_convs=None, temp_strides=None,
                    nonlocal_mod=1000, nonlocal_bn=True):

        if use_temp_convs is None:
            use_temp_convs = np.zeros(blocks).astype(int)
        if temp_strides is None:
            temp_strides = np.ones(blocks).astype(int)
        if len(use_temp_convs) < blocks:
            for _ in range(blocks - len(use_temp_convs)):
                use_temp_convs.append(0)
                temp_strides.append(1)

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1,
                          stride=[temp_strides[0], stride, stride],
                          padding=0,
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []

        for i in range(blocks):
            if i == 0:
                layers.append((str(i), block(self.inplanes, planes, stride, downsample, group=group,
                                             use_temp_conv=use_temp_convs[i], temp_stride=temp_strides[i])))
                self.inplanes = planes * block.expansion
            else:
                layers.append((str(i), block(self.inplanes, planes, group=group,
                                             use_temp_conv=use_temp_convs[i], temp_stride=temp_strides[i])))
            if i % nonlocal_mod == nonlocal_mod - 1:
                layers.append(
                    ('nl{}'.format(i), get_nonlocal_block(self.nltype)(self.inplanes, k=self.k, tk=self.tk, ts=self.ts, dropout=self.nl_drop)))
                print('add {} after res_block {} with {} planes'.format(
                    self.nltype, i, self.inplanes))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.temp_pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = nn.functional.avg_pool3d(x, [x.size(2), 7, 7], stride=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.mean(dim=(2, 3, 4))

        return x


# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model


# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model


def c2d_resnet34(**kwargs):

    use_temp_convs_1 = [0]
    temp_strides_1 = [1]
    use_temp_convs_2 = [0, 0, 0]
    temp_strides_2 = [1, 1, 1]
    use_temp_convs_3 = [0, 0, 0, 0]
    temp_strides_3 = [1, 1, 1, 1]
    use_temp_convs_4 = [0, ] * 6
    temp_strides_4 = [1, ] * 6
    use_temp_convs_5 = [0, 0, 0]
    temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(BasicBlock, [3, 4, 6, 3],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def c2d_resnet50(**kwargs):

    use_temp_convs_1 = [0]
    temp_strides_1 = [1]
    use_temp_convs_2 = [0, 0, 0]
    temp_strides_2 = [1, 1, 1]
    use_temp_convs_3 = [0, 0, 0, 0]
    temp_strides_3 = [1, 1, 1, 1]
    use_temp_convs_4 = [0, ] * 6
    temp_strides_4 = [1, ] * 6
    use_temp_convs_5 = [0, 0, 0]
    temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def i3d_resnet18(**kwargs):

    use_temp_convs_1 = [2]
    temp_strides_1 = [1]
    use_temp_convs_2 = [1] * 2
    temp_strides_2 = [1] * 2
    use_temp_convs_3 = [1] * 2
    temp_strides_3 = [1] * 2
    use_temp_convs_4 = [1] * 2
    temp_strides_4 = [1] * 2
    use_temp_convs_5 = [1] * 2
    temp_strides_5 = [1] * 2

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def i3d_resnet50(**kwargs):

    use_temp_convs_1 = [2]
    temp_strides_1 = [1]
    use_temp_convs_2 = [1, 1, 1]
    temp_strides_2 = [1, 1, 1]
    use_temp_convs_3 = [1, 0, 1, 0]
    temp_strides_3 = [1, 1, 1, 1]
    use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
    temp_strides_4 = [1, 1, 1, 1, 1, 1]
    use_temp_convs_5 = [0, 1, 0]
    temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def i3d_resnet50(**kwargs):

    use_temp_convs_1 = [2]
    temp_strides_1 = [1]
    use_temp_convs_2 = [1, 1, 1]
    temp_strides_2 = [1, 1, 1]
    use_temp_convs_3 = [1, 0, 1, 0]
    temp_strides_3 = [1, 1, 1, 1]
    use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
    temp_strides_4 = [1, 1, 1, 1, 1, 1]
    use_temp_convs_5 = [0, 1, 0]
    temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def c2d_resnet101(**kwargs):
    use_temp_convs_1 = [0]
    temp_strides_1 = [1]
    use_temp_convs_2 = [0, 0, 0]
    temp_strides_2 = [1, 1, 1]
    use_temp_convs_3 = [0, 0, 0, 0]
    temp_strides_3 = [1, 1, 1, 1]
    use_temp_convs_4 = [0, ] * 23
    temp_strides_4 = [1, ] * 23
    use_temp_convs_5 = [0, 0, 0]
    temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def i3d_resnet101(**kwargs):
    use_temp_convs_1 = [2]
    temp_strides_1 = [1]
    use_temp_convs_2 = [1, 1, 1]
    temp_strides_2 = [1, 1, 1]
    use_temp_convs_3 = [1, 0, 1, 0]
    temp_strides_3 = [1, 1, 1, 1]
    use_temp_convs_4 = []
    for i in range(23):
        if i % 2 == 0:
            use_temp_convs_4.append(1)
        else:
            use_temp_convs_4.append(0)

    temp_strides_4 = [1, ] * 23
    use_temp_convs_5 = [0, 1, 0]
    temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2,
                          use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2,
                        temp_strides_3, temp_strides_4, temp_strides_5]
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   use_temp_convs_set, temp_strides_set, **kwargs)
    return model


# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     return model


# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     return model

if __name__ == '__main__':
    import os
    import torch

    gpus = [0, 1]
    batch_size = 2 * len(gpus)

    video_len = 16
    nonlocal_mod = 2
    c2d = c2d_resnet50(nonlocal_mod=nonlocal_mod)
    c2d = torch.nn.DataParallel(c2d, device_ids=gpus).cuda()
    c2d = c2d.eval()
    data = torch.autograd.Variable(
        torch.rand(batch_size, 3, video_len, 224, 224))
    out = c2d(data)
    print(out.size())

    c2d = c2d_resnet50()
    c2d = torch.nn.DataParallel(c2d, device_ids=gpus).cuda()
    c2d = c2d.eval()
    data = torch.autograd.Variable(
        torch.rand(batch_size, 3, video_len, 256, 256))
    out = c2d(data)
    print(out.size())

    i3d = i3d_resnet50(nonlocal_mod=nonlocal_mod)
    i3d = torch.nn.DataParallel(i3d, device_ids=gpus).cuda()
    i3d = i3d.eval()
    data = torch.autograd.Variable(
        torch.rand(batch_size, 3, video_len, 224, 224))
    out = i3d(data)
    print(out.size())
