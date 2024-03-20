# CUDA_VISIBLE_DEVICES=1 nohup python main.py ../data/mycoco/fish_coco/train > LOG1.TXT &

# CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py '/data/home/haoyue/barlowtwins-main/checkpoint/nihpre/resnet50_1110.pth' --lr-classifier 0.3 > LOG.TXT &
# CUDA_VISIBLE_DEVICES=1 python evaluate.py '/data/home/haoyue/barlowtwins-main/checkpoint/nihpre/resnet50_1110.pth' --lr-classifier 0.3
# CUDA_VISIBLE_DEVICES=1 python evaluate.py '/data/home/haoyue/barlowtwins-main/checkpoint/nihpre/resnet50_1110.pth' --lr-classifier 0.1 --barlow_twins 'True'


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import models, datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torchvision
from sklearn.metrics import roc_auc_score

from dataset import ChestXrayDataset
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
# parser.add_argument('data', type=Path, metavar='DIR',
#                     help='path to dataset')
parser.add_argument('pretrained', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('--barlow_twins', default='True', type=str,
                    help='barlowtwins or imagenet resnet weights')
parser.add_argument('--weights', default='finetune', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train_percent', default=10, type=int,
                    help='size of traing set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.5, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint_1219/finetune_imagnet1_', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--random-seed', type=int, default=None,
                    help='Random seed for reproducibility')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
                                   
def main():
    args = parser.parse_args()
    # if args.train_percent in {1, 10}:
    #     args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt').readlines()
    args.ngpus_per_node = torch.cuda.device_count()
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    if 'SLURM_JOB_ID' in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    print(args.barlow_twins)
    if args.barlow_twins == 'True':
        print('load pretrained weight from barlow twins')
        model = models.resnet50()
        num_labels = 15
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_labels)
        model = model.cuda(gpu)
        state_dict = torch.load(args.pretrained, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    elif args.barlow_twins == 'False':
        print('load pretrained weight ImageNet')
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_labels = 15
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_labels)
        model = model.cuda(gpu)

    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    #criterion = nn.CrossEntropyLoss().cuda(gpu)
    criterion = nn.BCEWithLogitsLoss().cuda(gpu)
    

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    #traindir = args.data / 'train'
    #valdir = args.data / 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    '''
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    '''
    # Set up image transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    img_dir = '/data/public_data/NIH/images-224/images-224'
    csv_file = '/data/public_data/NIH/Data_Entry_2017.csv'
    test_list_file = '/data/public_data/NIH/test_list_NIH.txt'
    # Create training and validation
    if args.train_percent < 100:
        train_dataset = ChestXrayDataset(img_dir, csv_file, test_list_file, 
                                         split_type='train', transform=transform_train, select_percent=args.train_percent)
    else:
        train_dataset = ChestXrayDataset(img_dir, csv_file, test_list_file, 
                                         split_type='train', transform=transform_train)
    val_dataset = ChestXrayDataset(img_dir, csv_file, test_list_file, 
                                   split_type='test', transform=transform_test, select_percent=100)
    
    '''
    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls]))
    '''
    

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    start_time = time.time()
    best_auc = 0
    label_names = train_dataset.classes
    print(args.weights)
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        # evaluate
        model.eval()
        if args.rank == 0:
            all_outputs = []
            all_targets = []

            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    all_outputs.append(output)
                    all_targets.append(target.cuda(gpu, non_blocking=True))

            # Combine all outputs and targets into one tensor
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Calculate AUC
            label_auc_scores, average_auc = calculate_auc(all_outputs, all_targets, label_names)
            print(f"AUC Score: {average_auc}")
            print(f"AUC Score: {label_auc_scores}")

            if average_auc > best_auc:
                best_auc = average_auc
            print(f"Best AUC Score so far: {best_auc}")

            stats = dict(epoch=epoch, label_auc_scores=label_auc_scores, average_auc=average_auc)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)
            '''
            top1 = AverageMeter('Acc@1')
            top5 = AverageMeter('Acc@5')
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)
            '''
            
            

        # sanity check
        # if args.weights == 'freeze':
        #     reference_state_dict = torch.load(args.pretrained, map_location='cpu')
        #     model_state_dict = model.module.state_dict()
        #     for k in reference_state_dict:
        #         assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calculate_auc(outputs, targets, label_names):
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    print(outputs)
    print(targets)
    label_auc_scores = {}

    for i, label in enumerate(label_names):
        if len(np.unique(targets[:, i])) == 2:
            auc_score = roc_auc_score(targets[:, i], outputs[:, i])
            label_auc_scores[label] = auc_score
    average_auc = np.mean(list(label_auc_scores.values())) if label_auc_scores else float('nan')

    return label_auc_scores, average_auc

if __name__ == '__main__':
    main()
