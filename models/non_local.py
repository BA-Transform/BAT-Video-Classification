import torch
from torch import nn
from torch.nn import functional as F


def get_nonlocal_block(block_type):
    block_dict = {'nl3d': NONLocalBlock3D,
                  'nl2d': NONLocalBlock2D, 'bat': BATBlock}
    if block_type in block_dict:
        return block_dict[block_type]
    else:
        raise ValueError("UNKOWN NONLOCAL BLOCK TYPE:", block_type)


def _init_conv(conv_layer, std=0.01, zero_init=False):
    if zero_init:
        nn.init.constant_(conv_layer.weight, 0.0)
    else:
        nn.init.normal_(conv_layer.weight, 0.0, std)

    if len(list(conv_layer.parameters())) > 1:
        nn.init.constant_(conv_layer.bias, 0.0)


class NonLocalModule(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(NonLocalModule, self).__init__()

    def init_modules(self, zero_bn=True):
        bn_w = 0. if zero_bn else 1.
        for name, m in self.named_modules():
            if len(m._modules) > 0:
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, bn_w)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, bn_w)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif len(list(m.parameters())) > 0:
                raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", name, m)


class _NonLocalBlockND(NonLocalModule):

    def __init__(self, in_channels, inter_channels=None, dimension=3, head=1, sub_sample=True, bn_layer=True, use_scale=True, zero_init=False, **kwargs):
        super(_NonLocalBlockND, self).__init__(in_channels)

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.head = head
        self.sub_sample = sub_sample
        self.use_scale = use_scale
        self.zero_init = zero_init

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        assert self.inter_channels % self.head == 0

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            kernel_size = 2 if dimension <= 2 else [1, 2, 2]
            self.g = nn.Sequential(max_pool(kernel_size=kernel_size), self.g)
            self.phi = nn.Sequential(
                max_pool(kernel_size=kernel_size), self.phi)

    def init_modules(self):
        super().init_modules()
        if isinstance(self.W, nn.Sequential):
            _init_conv(self.W[0], zero_init=self.zero_init)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            _init_conv(self.W, zero_init=self.zero_init)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size * self.head,
                             self.inter_channels // self.head, -1)
        g_x = g_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(x).view(batch_size * self.head,
                                     self.inter_channels // self.head, -1)
        theta_x = theta_x.permute(0, 2, 1).contiguous()
        phi_x = self.phi(x).view(batch_size * self.head,
                                 self.inter_channels // self.head, -1)
        f = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            f = f * ((self.inter_channels // self.head)**-.5)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):

    def __init__(self, in_channels, **kwargs):
        super(NONLocalBlock1D, self).__init__(
            in_channels, dimension=1, **kwargs)


class NONLocalBlock2D(_NonLocalBlockND):

    def __init__(self, in_channels, **kwargs):
        super(NONLocalBlock2D, self).__init__(
            in_channels, dimension=2, **kwargs)


class NONLocalBlock3D(_NonLocalBlockND):

    def __init__(self, in_channels, **kwargs):
        super(NONLocalBlock3D, self).__init__(
            in_channels, dimension=3, **kwargs)


class BATransform(nn.Module):

    def __init__(self, in_channels, s, ts, k, tk):
        super(BATransform, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, k, 1),
                                   nn.BatchNorm3d(k),
                                   nn.ReLU(inplace=True))
        self.conv_p = nn.Conv3d(k, s * s * k, [1, s, 1])
        self.conv_q = nn.Conv3d(k, s * s * k, [1, 1, s])
        if tk > 0:
            self.conv_t = nn.Conv3d(k, ts * ts * tk, [ts, 1, 1])
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 1),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True))
        self.s = s
        self.ts = ts
        self.k = k
        self.tk = tk
        self.in_channels = in_channels

    def extra_repr(self):
        return 'BATransform({in_channels}, ts={ts}, s={s}, k={k}, tk={tk})'.format(**self.__dict__)

    def resize_mat(self, x, t):
        n, c, st, s, s1 = x.shape
        assert s == s1
        if t <= 1:
            return x
        x = x.view(n * c * st, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(n * c * st, s, s, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(n, c, st, s * t, s * t)
        return x

    def forward(self, x):
        scale_times = x.size(3) // self.s
        matrix_size = x.size(3) // scale_times
        out = self.conv1(x)
        n, _, t, h, w = out.size()
        rp = F.adaptive_max_pool3d(out, (t, matrix_size, 1))
        cp = F.adaptive_max_pool3d(out, (t, 1, matrix_size))
        if matrix_size == self.s:
            p = self.conv_p(rp).view(n, self.k, self.s, self.s, t)
            q = self.conv_q(cp).view(n, self.k, self.s, self.s, t)
        else:
            ones = x.new_ones(
                (1, 1, matrix_size, matrix_size, 1), requires_grad=False)
            p = x.new_zeros(n, self.k, matrix_size, matrix_size, t)
            p_out = self.conv_p(rp).view(n, self.k, self.s, self.s, t, -1)
            count = x.new_zeros(
                (1, 1, matrix_size, matrix_size, 1), requires_grad=False)
            for i in range(p_out.size(5)):
                p[:, :, i:self.s + i, i:self.s + i, :] += p_out[:, :, :, :, :, i]
                count[:, :, i:self.s + i, i:self.s + i, :] += 1
            count = torch.where(count > 0, count, ones)
            p /= count

            q = x.new_zeros(n, self.k, matrix_size, matrix_size, t)
            q_out = self.conv_q(cp).view(n, self.k, self.s, self.s, t, 2)
            count = x.new_zeros(
                (1, 1, matrix_size, matrix_size, 1), requires_grad=False)
            for i in range(q_out.size(5)):
                q[:, :, i:self.s + i, i:self.s + i, :] += q_out[:, :, :, :, :, i]
                count[:, :, i:self.s + i, i:self.s + i, :] += 1
            count = torch.where(count > 0, count, ones)
            q /= count
        p = F.softmax(p, dim=3)
        q = F.softmax(q, dim=2)
        p = p.view(n, self.k, 1, matrix_size, matrix_size, t).expand(
            n, self.k, x.size(1) // self.k, matrix_size, matrix_size, t).contiguous()
        p = p.view(n, x.size(1), matrix_size, matrix_size, t).permute(
            0, 1, 4, 2, 3).contiguous()
        q = q.view(n, self.k, 1, matrix_size, matrix_size, t).expand(
            n, self.k, x.size(1) // self.k, matrix_size, matrix_size, t).contiguous()
        q = q.view(n, x.size(1), matrix_size, matrix_size, t).permute(
            0, 1, 4, 2, 3).contiguous()
        p = self.resize_mat(p, h // matrix_size)
        q = self.resize_mat(q, w // matrix_size)
        y = p.matmul(x)
        y = y.matmul(q)
        if self.tk > 0:
            tp = F.adaptive_avg_pool3d(out, (self.ts, 1, 1))
            tm = self.conv_t(tp).view(n, self.tk, self.ts, self.ts)
            tm = F.softmax(tm, dim=3)
            tm = tm.view(n, self.tk, 1, 1, 1, self.ts, self.ts).expand(
                n, self.tk, x.size(1) // self.tk, h, w, self.ts, self.ts).contiguous()
            tm = tm.view(n, x.size(1), h * w, self.ts, self.ts)
            tm = self.resize_mat(tm, t // self.ts)
            tm = tm.view(n, x.size(1), h, w, t, t)
            y = y.permute(0, 1, 3, 4, 2).contiguous().view(
                n, x.size(1), h, w, t, 1)
            y = tm.matmul(y).squeeze(-1).permute(0, 1, 4, 2, 3).contiguous()

        y = self.conv2(y)

        return y


class BATBlock(NonLocalModule):

    def __init__(self, in_channels, r=2, s=7, ts=4, k=4, tk=1, dropout=0.2, **kwargs):
        super().__init__(in_channels)

        inter_channels = in_channels // r
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 1),
                                   nn.BatchNorm3d(inter_channels),
                                   nn.ReLU(inplace=True))
        self.batransform = BATransform(
            inter_channels, s, ts, k, tk)
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, in_channels, 1),
                                   nn.BatchNorm3d(in_channels),
                                   nn.ReLU(inplace=True))
        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.batransform(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x

    def init_modules(self):
        super().init_modules(zero_bn=False)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    sub_sample = False
    gpus = [0, 1]

    img = Variable(torch.rand(2, 4, 5))
    net = NONLocalBlock1D(4, sub_sample=sub_sample, bn_layer=False)
    net = torch.nn.DataParallel(net, device_ids=gpus).cuda()
    out = net(img)
    print(out.size())

    img = Variable(torch.rand(2, 4, 5, 3))
    net = NONLocalBlock2D(4, sub_sample=sub_sample)
    net = torch.nn.DataParallel(net, device_ids=gpus).cuda()
    out = net(img)
    print(out.size())

    img = Variable(torch.rand(2, 4, 5, 4, 5))
    net = NONLocalBlock3D(4, sub_sample=sub_sample)
    net = torch.nn.DataParallel(net, device_ids=gpus).cuda()
    out = net(img)
    print(out.size())
