import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from os import path
import json


def get_module(m, name):
    ns = name.split('.')
    for n in ns:
        m = getattr(m, n)
    return m


def set_module(m, name, new_m):
    ns = name.rsplit('.', 1)
    if len(ns) == 1:
        setattr(m, ns[0], new_m)
    else:
        m = get_module(m, ns[0])
        setattr(m, ns[1], new_m)


class LRAdjuster(object):

    def __init__(self, init_lr, lr_steps=[-1], higher=True, p=0.5, p_decay=0.8, config_file=None):
        self.init_lr = init_lr
        self.lr_steps = lr_steps
        self.higher = higher
        self.init_p = p
        self.p_decay = p_decay
        self.config_file = config_file
        self.flag = False
        self._init_param()
        if path.exists(self.config_file):
            self.load_config()
        else:
            self.save_config()

    def _init_param(self):
        self.lr = self.init_lr[0]
        self.c0 = 0
        self.c1 = 0
        self.last_lr_score = 0
        self.last_score = 0
        self.p = self.init_p
        print('[*0] Learning Rate:', self.lr)

    def _update_lr_by_steps(self, epoch):
        if len(self.init_lr) == 1:
            decay = 0.1 ** (sum(epoch >= np.array(self.lr_steps)))
            self.lr = self.init_lr[0] * decay
        else:
            self.lr = self.init_lr[0]
            for cur_lr, step in zip(self.init_lr, self.lr_steps):
                if epoch < step:
                    break
                self.lr = cur_lr

    def update_lr(self, epoch, score):
        if self.lr_steps[0] > 0:
            self._update_lr_by_steps(epoch)
            return self.lr

        if not self.higher:
            score = -score

        self.load_config()
        d_step_score = score - self.last_score
        self.last_score = max(score, self.last_lr_score)
        if d_step_score < self.p:
            self.c1 += 1
            d_lr_score = score - self.last_lr_score
            if d_lr_score < self.p:
                self.c0 += 1
            if self.c0 >= 3:
                self._init_param()
            elif self.c1 >= 3:
                self.c1 = 0
                self.lr *= 0.1
                self.p *= self.p_decay
                if self.last_score - self.last_lr_score >= self.p:
                    self.c0 = max(self.c0 - 1, 0)
                self.last_lr_score = self.last_score
                print('[*1] Learning Rate:', self.lr)
        else:
            self.c1 = max(self.c1 - 1, 0)
        self.save_config()

        return self.lr

    def save_config(self):
        if self.config_file is None:
            return
        with open(self.config_file, 'w') as out:
            json.dump(vars(self), out)

    def load_config(self):
        if self.config_file is None:
            return
        with open(self.config_file) as f:
            state = json.load(f)
            for k, v in state.items():
                setattr(self, k, v)


def isBN(m):
    bns = [nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d]
    for bn in bns:
        if isinstance(m, bn):
            return True
    return False


def isConv(m):
    convs = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    for conv in convs:
        if isinstance(m, conv):
            return True
    return False


class PolicyUnit(object):

    def __init__(self, name, lr_mult, decay_mult):
        self.name = name
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult
        self.params = []
        self.name_list = []

    def add(self, p, p_name):
        self.params.append(p)
        self.name_list.append(p_name)

    def get_dict(self):
        return {'params': self.params,
                'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult,
                'name': self.name, 'name_list': self.name_list}

    def has_params(self):
        return len(self.params) > 0


class TrainPolicy(object):

    def __init__(self, prefix='', lr_mult=1, decay_mult=1, partial_bn=False, no_bn=False):
        self.partial_bn = partial_bn
        self.no_bn = no_bn
        self.bn_cnt = 0
        self.weight = PolicyUnit(
            prefix + 'weight', 1 * lr_mult, 1 * decay_mult)
        self.bias = PolicyUnit(prefix + 'bias', 2 * lr_mult, 0 * decay_mult)
        self.bn = PolicyUnit(prefix + 'bn', 1 * lr_mult, 0 * decay_mult)

    def add_module(self, m, name):
        if len(m._modules) > 0:
            return
        ps = list(m.parameters())
        if isConv(m):
            self.weight.add(ps[0], name)
            if len(ps) == 2:
                self.bias.add(ps[1], name)
        elif isinstance(m, torch.nn.Linear):
            self.weight.add(ps[0], name)
            if len(ps) == 2:
                self.bias.add(ps[1], name)
        elif isBN(m):
            self.bn_cnt += 1
            if self.partial_bn and self.bn_cnt > 1:
                return
            elif self.no_bn:
                return
            self.bn.add(ps[0], name + '.weight')
            self.bn.add(ps[1], name + '.bias')
        elif len(list(m.parameters())) > 0:
            raise ValueError(
                "New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    def get_policy(self):
        policy = []
        policy_units = [self.weight, self.bias, self.bn]
        for u in policy_units:
            if u.has_params():
                policy.append(u.get_dict())
        return policy


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_policy(model):
    policy = TrainPolicy()
    for name, m in model.named_modules():
        policy.add_module(m, name)
    return policy.get_policy()


def load_checkpoint(model, checkpoint, strict=True):
    state = model.state_dict()
    missing = []
    diff = []
    new_checkpoint = {}
    for k in checkpoint:
        if k.startswith('module.'):
            k_new = k[len('module.'):]
            new_checkpoint[k_new] = checkpoint[k]
        else:
            new_checkpoint[k] = checkpoint[k]
    checkpoint = new_checkpoint
    for k in state:
        if k not in checkpoint:
            missing.append(k)
            if 'num_batches_tracked' not in k:
                print('MISSING', k)
        else:
            v = checkpoint[k]
            shape1 = state[k].shape
            shape2 = checkpoint[k].shape
            if shape1 != shape2:
                if len(shape1) == 5 and len(shape2) == 4:  # inflate 2dconv to 3dconv
                    assert (shape1[:2] == shape2[:2]
                            and shape1[-2:] == shape2[-2:])
                    t = shape1[2]
                    v = torch.stack([v] * t, dim=2) / t
                    if t > 1:
                        print('INFLATE {} {:d} times.'.format(
                            k, t), shape2, '->', shape1)
                    assert v.shape == shape1
                    state[k] = v
                else:
                    diff.append(k)
                    print('DIFFERENT SHAPE', k, shape1, shape2)
            else:
                state[k] = v

    if strict:
        assert(len(missing) == 0 and len(diff) == 0)
    model.load_state_dict(state)


def affine(model):
    for m in model.modules():
        if isBN(m):
            m.track_running_stats = False


def freeze_bn(model):
    for m in model.modules():
        if isBN(m):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
