import argparse
import models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)
parser.add_argument('--root_path', default="", type=str)
parser.add_argument('--log_dir', default='log', type=str)


# ========================= Dataset Configs ==========================
parser.add_argument('--num_segments', default=1, type=int)
parser.add_argument('--seq_length', default=8, type=int)
parser.add_argument('--sample_rate', default=8, type=int,
                    help='video sample rate')
parser.add_argument('--read_mode', default='img',
                    choices=['img', 'video'])
parser.add_argument('--num_classes', default=400, type=int)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="C2D-ResNet50",
                    choices=model_names)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--nonlocal_mod', default=[1000], type=int, nargs="+")
parser.add_argument('--nltype', default='nl3d', type=str)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--tk', default=0, type=int)
parser.add_argument('--ts', default=4, type=int)
parser.add_argument('--nl_drop', default=0.2, type=float)
parser.add_argument('--freeze_bn', action='store_true')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--coslr', action='store_true')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--nesterov', action='store_true',
                    help='enables Nesterov momentum')
parser.add_argument('--use_affine', action='store_true', help='freeze BN')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save-freq', '-sf', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save-vg', help='save the summary of value and gradient',
                    default=False, action='store_true')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--soft_resume', action='store_true')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_dir', type=str, default="checkpoints")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--seed', default=-1, type=int, help='random seed')
