from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trn
import os
import numpy as np
import models.densenet as dn
import torchvision.models as models
import models.resnet_train as resnet
from models.densenet_dice import DenseNet3
from models.wideresidual import wideresnet
from models.mobilenet import mobilenet
from utils.svhn_loader import SVHN
import torchvision.datasets as dset
from models.wrn import WideResNet
import pdb

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', type=str, choices=["cifar10", "cifar100", "imagenet"])
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "densenet161", "wide_resnet50_2", "mobilenet_v2", "wrn", "new_resnet50"])
parser.add_argument('--batch_size', type=int, default=100)

args = parser.parse_args()

if 'cifar' in args.dataset:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
elif 'imagenet' in args.dataset:
    mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
ood_transform=trn.Compose([
    trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
    trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)
])
eval_transform = trn.Compose([
    trn.Resize(32),
    trn.CenterCrop(32),
    trn.ToTensor(),
    trn.Normalize(mean, std)
])

class Densenet161(nn.Module):
    def __init__(self):
        super(Densenet161, self).__init__()
        self.net = models.densenet161(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

    def features(self, x):
        out = self.extractor(x)
        return out

class Wide_resnet50_2(nn.Module):
    def __init__(self):
        super(Wide_resnet50_2, self).__init__()
        self.net = models.wide_resnet50_2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])
        self.emb_extractor = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        out = self.extractor(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.emb_extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

    def features(self, x):
        out = self.emb_extractor(x)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])
        self.emb_extractor = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        out = self.extractor(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.emb_extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

    def features(self, x):
        out = self.emb_extractor(x)
        return out

class MobileNet_V2(nn.Module):
    def __init__(self):
        super(MobileNet_V2, self).__init__()
        self.net = models.mobilenet_v2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1], *list(self.net.children())[-1][:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1][-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = ash_s(out, 90)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

    def features(self, x):
        out = self.extractor(x)
        return out

args = parser.parse_args()
if args.dataset == 'cifar10':
    num_classes = 10
    if args.model == "resnet50":
        model = resnet.resnet50(num_classes=10)
        model.load_state_dict(torch.load("./ckpt/resnet50_cifar10-192-best-0.9546999931335449.pth"))
        featdim = 2048
    elif args.model == "densenet_dice":
        model = DenseNet3(100, 10)
        model.load_state_dict(torch.load("./ckpt/checkpoint_10.pth.tar")["state_dict"])
        featdim = 342
    elif args.model == "wideresnet":
        model = wideresnet(num_classes=10)
        model.load_state_dict(torch.load("./ckpt/wideresnet_cifar10_epoch195_acc0.960599958896637.pt", map_location='cuda:0'))
        featdim = 640
    elif args.model == "wrn":
        model = WideResNet(40, 10, 2, dropRate=0.3)
        model.load_state_dict(torch.load("./ckpt/wrn_cifar10_190_best_0.9469999670982361.pth"))
        featdim = 128
    elif args.model == "new_resnet50":
        model = resnet.resnet50(num_classes=10)
        model.load_state_dict(torch.load("./ckpt/new_resnet50_cifar10_180_0.9540999531745911.pth"))
        featdim = 2048
elif args.dataset == 'cifar100':
    num_classes = 100
    if args.model == "resnet50":
        model = resnet.resnet50(num_classes=100)
        model.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
        featdim = 2048
    elif args.model == "densenet_dice":
        model = DenseNet3(100, 100)
        model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
        featdim = 342
    elif args.model == "wideresnet":
        model = wideresnet(num_classes=100)
        model.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
        featdim = 640
    elif args.model == "wrn":
        model = WideResNet(40, 100, 2, dropRate=0.3)
        model.load_state_dict(torch.load("./ckpt/wrn_cifar100_190_best_0.7486000061035156.pth"))
        featdim = 128
    elif args.model == "new_resnet50":
        model = resnet.resnet50(num_classes=100)
        model.load_state_dict(torch.load("./ckpt/new_resnet50_cifar100_182_0.7894999980926514.pth"))
        featdim = 2048
elif args.dataset == 'imagenet':
    num_classes = 1000
    if args.model == "resnet50":
        model = ResNet50()
        featdim = 2048
    elif args.model == "densenet161":
        model = Densenet161()
        featdim = 2208
    elif args.model == "wide_resnet50_2":
        model = Wide_resnet50_2()
        featdim = 2048
    elif args.model == "mobilenet_v2":
        model = MobileNet_V2()
        featdim = 1280
    model = model.cuda()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
if args.dataset in {'cifar10', 'cifar100'}:
    transform_test = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])
    if args.dataset == "cifar10":
        trainset = dset.CIFAR10("../data/cifar10", train=True, transform=train_transform, download=False)
    elif args.dataset == "cifar100":
        trainset = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    id_train_size = 50000

    cache_name = f"./cache/{args.dataset}_{args.model}_in.npy"
    if not os.path.exists(cache_name):
        feat_log = np.zeros((id_train_size, featdim))
        score_log = np.zeros((id_train_size, num_classes))
        label_log = np.zeros(id_train_size)

        model.eval()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ind = batch_idx * args.batch_size
            end_ind = min((batch_idx + 1) * args.batch_size, len(trainset))

            outputs = model.features(inputs)
            out = F.adaptive_avg_pool2d(outputs, 1)
            out = out.view(out.size(0), -1)
            if args.model == "densenet_dice" or args.model == "resnet50" or args.model == "wrn" or args.model == "new_resnet50":
                score = model.fc(out)
            elif args.model == "wideresnet":
                score = model.linear(out)
            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 10 == 0:
                print(batch_idx)
        np.save(cache_name, (feat_log.T, score_log.T, label_log))
    else:
        feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log, score_log = feat_log.T, score_log.T

    np.save(f"cache/{args.dataset}_{args.model}_feat_stat.npy", feat_log.mean(0))
    print("done")
else:
    transform_train_largescale = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
    transform_test_largescale = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('../data/1_of_10_train', transform_train_largescale),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('../data/val', transform_test_largescale),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    id_train_size = 121706

    feat_log = np.zeros((id_train_size, featdim))
    score_log = np.zeros((id_train_size, num_classes))
    label_log = np.zeros(id_train_size)

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        start_ind = batch_idx * args.batch_size
        end_ind = min((batch_idx + 1) * args.batch_size, len(trainloader.dataset))
        print("batch_idx: %d" % (batch_idx))
        outputs = model.features(inputs)
        out = F.adaptive_avg_pool2d(outputs, 1)
        out = out.view(out.size(0), -1)
        score = model.fc(out)
        feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
        label_log[start_ind:end_ind] = targets.data.cpu().numpy()
        score_log[start_ind:end_ind] = score.data.cpu().numpy()
        if batch_idx % 10 == 0:
            print(f"{batch_idx}/{len(trainloader)}")


    np.save(f"./cache/{args.dataset}_{args.model}_feat_stat.npy", feat_log.mean(0))
    print("done")
