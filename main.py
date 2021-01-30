import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
import numpy as np
import random

from dataset import TSNDataSet
from transforms import *
from opts import parser
from utils import *
import models


best_prec1 = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.channel = 3
    args.img_size = 224
    check_rootfolders(args)
    args.snapshot_pref = os.path.join(args.snapshot_dir, args.arch)
    with open(os.path.join(args.log_dir, 'params.txt'), 'w') as out:
        print(args, file=out)

    if args.seed < 0:
        args.seed = np.random.randint(1000000)

    print('seed:', args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # model
    model = models.__dict__[args.arch](
        num_classes=args.num_classes, dropout=args.dropout,
        nonlocal_mod=args.nonlocal_mod, nltype=args.nltype,
        k=args.k, tk=args.tk, ts=args.ts, nl_drop=args.nl_drop)
    print(model)

    cudnn.benchmark = True

    # dataloader
    input_mean = [103.939, 116.779, 123.68]
    input_std = [1]
    normalize = GroupNormalize(input_mean, input_std)

    train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                                         GroupRandomHorizontalFlip(is_flow=False)])
    roll_flag = False if args.read_mode == 'video' else True
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=args.seq_length,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=roll_flag),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ]), read_mode=args.read_mode, skip=args.sample_rate - 1),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=args.seq_length,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(256),
                       GroupCenterCrop(224),
                       Stack(roll=roll_flag),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ]), read_mode=args.read_mode, skip=args.sample_rate - 1),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    policies = get_policy(model)
    if args.use_affine:
        affine(model)
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    train_writer = SummaryWriter(os.path.join(args.log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(args.log_dir, 'val'))

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            if 'state_dict' in checkpoint:
                args.start_epoch = checkpoint[
                    'epoch'] if 'epoch' in checkpoint else 0
                best_prec1 = checkpoint[
                    'best_prec1'] if 'best_prec1' in checkpoint else 0
                load_checkpoint(model, checkpoint[
                                'state_dict'], strict=not args.soft_resume)
                if 'optim' in checkpoint:
                    print('=> loading optimizer')
                    try:
                        optimizer.load_state_dict(checkpoint['optim'])
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if torch.is_tensor(v):
                                    state[k] = v.cuda()
                    except:
                        print('[WARNING!]: loading optimizer error')
                if args.soft_resume:
                    args.start_epoch = 0
                    best_prec1 = 0
                print(("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.evaluate, checkpoint['epoch'])))
            else:
                load_checkpoint(model, checkpoint, strict=not args.soft_resume)
                print("=> initialize model from checkpoint")
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    model = torch.nn.DataParallel(model).to(device)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, val_writer)
        return

    score = 0
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        score = train(train_loader, model, criterion,
                      optimizer, epoch, train_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) >= args.epochs * 0.9:
            prec1 = validate(val_loader, model, criterion,
                             epoch + 1, val_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_prec1,
                'optim': optimizer.state_dict(),
            }, is_best)

        elif (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_prec1,
                'optim': optimizer.state_dict(),
            }, False)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    if args.freeze_bn:
        print('Freeze All BN Layers')
        freeze_bn(model)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)
        input = input.view(-1, args.seq_length, args.channel, args.img_size,
                           args.img_size).permute(0, 2, 1, 3, 4).contiguous()

        step = epoch * len(train_loader) + i
        adjust_learning_rate(optimizer, epoch, step, len(train_loader), args)
        # compute output
        output = model(input)
        output = tsn(output)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(
                model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(
                    total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and (i % args.print_freq == 0 or i + 1 == len(train_loader)):
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

            writer.add_scalar('loss', losses.avg, step)
            writer.add_scalar('pre@1', top1.avg, step)
            writer.add_scalar('pre@5', top5.avg, step)
            writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], step)
            if args.save_vg:
                params = optimizer.param_groups
                for param in params:
                    for ii, p in enumerate(param['params']):
                        writer.add_scalar('grad/' + param['name_list'][ii] + '_' + param['name'],
                                          p.grad.detach().abs().mean().cpu().numpy(), step)
                        writer.add_scalar('value/' + param['name_list'][ii] + '_' + param['name'],
                                          p.detach().abs().mean().cpu().numpy(), step)

    return top1.avg


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            input = input.view(-1, args.seq_length, args.channel, args.img_size,
                               args.img_size).permute(0, 2, 1, 3, 4).contiguous()

            # compute output
            output = model(input)
            output = tsn(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i > 0 and (i % args.print_freq == 0 or i + 1 == len(val_loader)):
                step = epoch * len(val_loader) + i
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5)))

                writer.add_scalar('loss', losses.avg, step)
                writer.add_scalar('pre@1', top1.avg, step)
                writer.add_scalar('pre@5', top5.avg, step)

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def check_rootfolders(args):
    timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    args.snapshot_dir = path.join(args.snapshot_dir, timestamp)
    args.log_dir = path.join(args.log_dir, timestamp)
    if not path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join(
            (args.snapshot_pref, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


def tsn(output):
    if args.num_segments <= 1:
        return output
    output = output.view((-1, args.num_segments) + output.size()[1:])
    output = output.mean(dim=1)
    return output


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """Warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * float(step) / (args.warmup_epochs * len_epoch)

    elif args.coslr:
        nmax = len_epoch * args.epochs
        lr = args.lr * 0.5 * (np.cos(step / nmax * np.pi) + 1)
    else:
        decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
        lr = args.lr * decay

    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
