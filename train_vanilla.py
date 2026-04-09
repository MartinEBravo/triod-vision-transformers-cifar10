"""Vanilla ResNet18 on CIFAR-10 baseline (no TriOD)."""
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time

from models.resnet import ResNet18
from utils import progress_bar
from randomaug import RandAugment

parser = argparse.ArgumentParser(description='Vanilla ResNet18 CIFAR-10 Baseline')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--noaug', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
if args.dataset == 'cifar10':
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    num_classes = 10
    dataset_class = torchvision.datasets.CIFAR10
elif args.dataset == 'cifar100':
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    num_classes = 100
    dataset_class = torchvision.datasets.CIFAR100

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if not args.noaug:
    transform_train.transforms.insert(0, RandAugment(2, 14))

trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=8)
testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# Model
net = ResNet18(num_classes=num_classes).to(device)

decay, no_decay = [], []
for name, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() == 1 or name.endswith(".bias"):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = optim.SGD([
    {"params": decay, "weight_decay": args.weight_decay},
    {"params": no_decay, "weight_decay": 0},
], lr=args.lr, momentum=0.9)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

best_acc = 0

def train(epoch):
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss/(batch_idx+1)))
    return train_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader))
    acc = 100. * correct / total
    test_loss = test_loss / (batch_idx + 1)
    print("Val Loss: {:.3f} | Val Acc: {:.3f}%".format(test_loss, acc))
    if acc > best_acc:
        print('Saving..')
        state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vanilla-res18-{}-ckpt.t7'.format(args.dataset))
        best_acc = acc
    return test_loss, acc

for epoch in range(args.n_epochs):
    start = time.time()
    print('\nEpoch: %d' % epoch)
    train_loss = train(epoch)
    val_loss, val_acc = test(epoch)
    scheduler.step()
    print("Epoch {} | Train Loss: {:.3f} | Val Loss: {:.3f} | Val Acc: {:.3f}% | Best: {:.3f}% | Time: {:.1f}s".format(
        epoch, train_loss, val_loss, val_acc, best_acc, time.time()-start))

print("Training completed. Best acc: {:.3f}%".format(best_acc))
