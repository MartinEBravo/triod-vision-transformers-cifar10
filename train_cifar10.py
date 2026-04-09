# -*- coding: utf-8 -*-
'''
Train CIFAR10/CIFAR100 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
modified to support CIFAR100
'''

from __future__ import print_function
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import math

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import wandb

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs
from models.dyt import DyT

# Check triod prefix OD implementation
from triod.utils import compute_cum_outputs, test_prefix_od
from triod.wrapper import triod

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/ImageNet Training')
parser.add_argument('--lr', default=5e-2, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--project', default='triod', type=str, help='wandb project name')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='res18')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--gpu', default='all', type=str, help='GPU id to use.')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10, cifar100, imagenet)')
parser.add_argument('--seed', default=42, type=int, help='random seed')
# TriOD
parser.add_argument('--no-triangular', action='store_true', help='whether to use triangular or not')
parser.add_argument('--min_p', default=0.2, type=float, help='minimum p for TriOD models')
parser.add_argument('--n_models', default=5, type=int, help='number of models for TriOD models')
parser.add_argument('--kd_alpha_max', default=1.0, type=float, help='maximum kd alpha for DyT')
parser.add_argument('--kd_beta_max', default=0.0, type=float, help='maximum hkd beta for DyT')
parser.add_argument('--kd_gamma_max', default=0.0, type=float, help=' maximum tkd gamma for DyT')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizer')
parser.add_argument('--kd_constant', action='store_true', help='whether to use constant kl alpha or not')
parser.add_argument('--criterion', default='kl', type=str, help='criterion to use (kl, mse)')
parser.add_argument('--T', default=4.0, type=float, help='temperature for kd loss')

args = parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(args.seed)

# take in args
usewandb = not args.nowandb
triangular = not args.no_triangular


if usewandb:
    wandb.init(
        entity="martin-bravo-mbzuai",
        project=args.project,
        name="{}_lr{}_{}".format(args.net, args.lr, args.dataset)
    )
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = not args.noaug

print(f"use_amp: {use_amp}, aug: {aug}")
print(f"triangular: {triangular}, min_p: {args.min_p}, n_models: {args.n_models}")
input("Press Enter to continue...")


if not args.gpu=='all' and not args.dp:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
    dataset_class = torchvision.datasets.CIFAR10
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
    dataset_class = torchvision.datasets.CIFAR100
elif args.dataset=="imagenet":
    # TODO: implement hugging face dataset for imagenet and turn it into a pytorch dataset
    pass

else:
    raise ValueError("Dataset must be either 'cifar10', 'cifar100', or 'imagenet'")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2
    M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))

def seed_worker(worker_id):
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

# Prepare dataset
trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, generator=g, worker_init_fn=seed_worker)

testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, generator=g, worker_init_fn=seed_worker)

# Set up class names based on the dataset
if args.dataset == 'cifar10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    # CIFAR100 has 100 classes and Imagenet has 1000 classes,
    # so we don't list them all here
    classes = None

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18(num_classes=num_classes)
elif args.net=='vgg':
    net = VGG('VGG19', num_classes=num_classes)
elif args.net=='res34':
    net = ResNet34(num_classes=num_classes)
elif args.net=='res50':
    net = ResNet50(num_classes=num_classes)
elif args.net=='res101':
    net = ResNet101(num_classes=num_classes)
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = num_classes
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10/100
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="dyt":
    # DyT for cifar10/100
    net = DyT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=num_classes,
                downscaling_factors=(2,2,2,1))
elif args.net=="mobilevit":
    net = mobilevit_xxs(size, num_classes)
else:
    raise ValueError(f"'{args.net}' is not a valid model")

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

# triodify the model for prefix OD
p_s = np.linspace(args.min_p, 1.0, num=args.n_models)
net = triod(net, triangular=triangular).to(device) # move to device after triodifying
assert net.p_test(next(iter(trainloader))[0].to(device), p_s=p_s), "TriOD property test failed"

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

decay, no_decay = [], []
for name, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() == 1 or name.endswith(".bias"):
        no_decay.append(p)
    else:
        decay.append(p)
if args.opt == "adam":
    optimizer = optim.Adam([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0},
        ],
        lr=args.lr
    )
elif args.opt == "sgd":
    optimizer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=args.lr,
        momentum=0.9,
    )
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    if not args.kd_constant:
        cos_factor = (1 - math.cos(math.pi * epoch / args.n_epochs)) / 2.
        alpha, beta, gamma = (
            args.kd_alpha_max * cos_factor,
            args.kd_beta_max * cos_factor,
            args.kd_gamma_max * cos_factor,
        )
    else:
        alpha, beta, gamma = args.kd_alpha_max, args.kd_beta_max, args.kd_gamma_max

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            #####################################################
            ###########  Cross Entropy Loss #####################
            #####################################################
            # output = net(inputs, all_models=True)
            prelast = net(inputs, return_prelast=True)
            full_logits = net.classify(prelast)
            ce_loss = F.cross_entropy(full_logits, targets)
            #####################################################
            ###########  Knowledge Distillation Loss ############
            #####################################################
            kd_loss, hkd_loss, tkd_loss = 0.0, 0.0, 0.0
            if args.n_models > 1:
                prev_logits = None
                for i, logits_i in enumerate(compute_cum_outputs(prelast, net, p_s)):
                    # Knowledge Distillation, current logit is the student, teacher is full model
                    if alpha > 0.0:
                        if i == len(p_s) - 1:
                            pass
                        else:
                            if args.criterion == 'ce':
                                kd_loss = kd_loss + F.cross_entropy(
                                    logits_i, # Student is current logit
                                    F.softmax(full_logits, dim=-1).detach() # Teacher is the full model
                                )
                            elif args.criterion == 'kl':
                                kd_loss = kd_loss + F.kl_div(
                                    F.log_softmax(logits_i / args.T, dim=-1),
                                    F.softmax(full_logits / args.T, dim=-1).detach(),
                                    reduction='batchmean'
                                ) * (args.T * args.T)

                    # Hierarchical Knowledge Distillation, current logit is the teacher, previous is student
                    if beta > 0.0:
                        if prev_logits is None: # first iteration
                            pass
                        else: 
                            if args.criterion == 'ce':
                                hkd_loss = hkd_loss + F.cross_entropy(
                                    prev_logits, # Student is previous logit
                                    F.softmax(logits_i, dim=-1).detach() # Teacher is current logit
                                )
                            elif args.criterion == 'kl':
                                hkd_loss = hkd_loss + F.kl_div(
                                    F.log_softmax(prev_logits / args.T, dim=-1),
                                    F.softmax(logits_i / args.T, dim=-1).detach(),
                                    reduction='batchmean'
                                ) * (args.T * args.T)
                        prev_logits = logits_i # update for next iteration

                    # Targeted Knowledge Distillation, current logit is the student, teacher is ground truth
                    if gamma > 0.0:
                        if i == len(p_s) - 1:
                            pass
                        else:
                            tkd_loss = tkd_loss + F.cross_entropy(
                                logits_i,
                                targets
                            )

                kd_loss /= (len(p_s) - 1)
                hkd_loss /= (len(p_s) - 1)
                tkd_loss /= (len(p_s) - 1)
            loss = ce_loss + alpha * kd_loss + beta * hkd_loss + gamma * tkd_loss

        # Backward + step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    losses = np.zeros(len(p_s))
    accs = np.zeros(len(p_s))
    agreements = np.zeros(len(p_s))
    sum_min_agree_p = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            prelast = net(inputs, return_prelast=True)
            logits_full = net.classify(prelast)
            losses[-1] += F.cross_entropy(logits_full, targets).item()
            accs[-1] += (logits_full.argmax(dim=-1) == targets).sum().item()/targets.size(0)
            if args.n_models > 1:
                full_preds = logits_full.argmax(dim=-1)
                batch_agrees = []
                for i, logits_i in enumerate(compute_cum_outputs(prelast, net, p_s)):
                    preds_i = logits_i.argmax(dim=-1)
                    batch_agrees.append(preds_i.eq(full_preds))
                    agreements[i] += preds_i.eq(full_preds).sum().item() / targets.size(0)
                    if i == len(p_s) - 1:
                        continue
                    loss = F.cross_entropy(logits_i, targets)
                    losses[i] += loss.item()
                    _, predicted = logits_i.max(1)
                    accs[i] += predicted.eq(targets).sum().item()/targets.size(0)
                # Smallest p that agrees with the full model per sample
                agree_matrix = torch.stack(batch_agrees, dim=0)  # (n_models, batch_size)
                first_agree_idx = agree_matrix.float().argmax(dim=0).cpu().numpy()
                min_p_per_sample = p_s[first_agree_idx]
                # Where no model agrees, argmax returns 0; default to 1.0
                any_agree = agree_matrix.any(dim=0).cpu().numpy()
                min_p_per_sample[~any_agree] = 1.0
                sum_min_agree_p += min_p_per_sample.sum()
                n_samples += targets.size(0)
            progress_bar(batch_idx, len(testloader))

    accs = 100*np.array(accs)/(batch_idx+1)
    losses = losses/(batch_idx+1)
    agreements = 100 * agreements / (batch_idx + 1)
    avg_min_agree_p = sum_min_agree_p / n_samples if n_samples > 0 else 0.0
    for i, p in enumerate(p_s):
        print("p={:.2f} | Loss: {:.3f} | Acc: {:.3f}% | Agree: {:.1f}%".format(p, losses[i], accs[i], agreements[i]))
    print("Avg smallest agreeing p: {:.4f}".format(avg_min_agree_p))

    # Save checkpoint.
    acc = np.mean(accs)
    if acc > best_acc:
        print('Saving..')
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch))
        best_acc = acc
    return losses, accs, agreements, avg_min_agree_p

if usewandb:
    wandb.watch(net)
    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_losses, val_accs, val_agreements, avg_min_p = test(epoch)
    
    scheduler.step() # step cosine scheduling
    
    # Log training..
    if usewandb:
        log_payload = {
            'epoch': epoch,
            'train_loss': trainloss,
            'mean_loss': np.mean(val_losses),
            "mean_acc": np.mean(val_accs),
            "best_acc": best_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": args.weight_decay,
            "epoch_time": time.time()-start,

        }
        for i, p in enumerate(p_s):
            log_payload[f"acc_p_{p:.2f}"] = val_accs[i]
            log_payload[f"loss_p_{p:.2f}"] = val_losses[i]
            log_payload[f"agree_p_{p:.2f}"] = val_agreements[i]
        log_payload["avg_min_agree_p"] = avg_min_p
        wandb.log(log_payload)

print("Training completed. Best acc: {:.3f}".format(best_acc))

# writeout wandb
if usewandb:
    wandb.save("wandb_{}_{}.h5".format(args.net, args.dataset))