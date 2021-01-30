import argparse
import time

import numpy as np
import torch.nn.parallel
import torchvision
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
import models
from transforms import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="C2D-ResNet50",
                    choices=model_names)
parser.add_argument('--tsm', action='store_true')
parser.add_argument('--num_classes', default=400, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--nonlocal_mod', type=int, default=[1000], nargs="+")
parser.add_argument('--nltype', type=str, default='nl3d')
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--tk', default=0, type=int)
parser.add_argument('--ts', default=4, type=int)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--seq_length', default=32, type=int,
                    help='sequnce length, used for 3D convolution')
parser.add_argument('--sample_rate', default=2, type=int,
                    help='video sample rate')
parser.add_argument('--test_segments', default=1, type=int)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--root_path', default="", type=str)
parser.add_argument('--div', default=1, type=int,
                    help="divide the batch to smaller batches")
parser.add_argument('--read_mode', default='img',
                    choices=['img', 'video', 'h5'])

args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.__dict__[args.arch](
    num_classes=args.num_classes, nonlocal_mod=args.nonlocal_mod,
    k=args.k, tk=args.tk, ts=args.ts, nltype=args.nltype, tsm=args.tsm)
checkpoint = torch.load(args.weights)
if 'state_dict' in checkpoint:
    print("model epoch {} best prec@1: {}".format(
        checkpoint['epoch'], checkpoint['best_prec1']))

    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

scale_size = int(args.img_size / 224 * 256)
input_size = args.img_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, scale_size)
    ])
elif args.test_crops == 3:
    cropping = torchvision.transforms.Compose([
        GroupScale(size=scale_size),
        MultiCrop(scale=scale_size)
    ])
else:
    raise ValueError(
        "Only 1 and 10 and 3 crops are supported while we got {}".format(args.test_crops))

input_mean = [103.939, 116.779, 123.68]
input_std = [1]
normalize = GroupNormalize(input_mean, input_std)
roll_flag = False if args.read_mode == 'video' else True
data_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.test_list, num_segments=args.test_segments,
               new_length=args.seq_length,
               test_mode=True,
               transform=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=roll_flag),
                   ToTorchFormatTensor(div=False),
                   normalize,
               ]), read_mode=args.read_mode, skip=args.sample_rate - 1),
    batch_size=1, shuffle=False,
    num_workers=args.workers, pin_memory=True)

model = torch.nn.DataParallel(model).to(device)
model.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data

    c = 3
    input_var = data.to(device)
    input_var = input_var.view(-1, args.seq_length, c, data.size(2),
                               data.size(3)).permute(0, 2, 1, 3, 4).contiguous()

    batch_size = input_var.shape[0]
    if args.div > 1:
        assert batch_size % args.div == 0
        small_batch_size = batch_size // args.div
        rst = []
        for i in range(args.div):
            small_batch = input_var[
                i * small_batch_size:(i + 1) * small_batch_size]
            rst.append(model(small_batch).data.cpu().numpy().copy())
        rst = np.concatenate(rst, axis=0)
    else:
        rst = model(input_var).data.cpu().numpy().copy()
    return i, rst.reshape((batch_size, 1, args.num_classes)), label.item()


proc_start_time = time.time()

correct = 0
with torch.no_grad():
    for i, (data, label) in data_gen:
        rst = eval_video((i, data, label))
        output.append(rst[1:])
        cnt_time = time.time() - proc_start_time
        if np.argmax(np.mean(rst[1], axis=0)) == rst[2]:
            correct += 1
        print('video {} done, total {}/{}, average {} sec/video, acc {:.2f}'.format(i, i + 1,
                                                                                    total_num,
                                                                                    float(
                                                                                        cnt_time) / (i + 1),
                                                                                    correct / float(i + 1) * 100))

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e: i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)
