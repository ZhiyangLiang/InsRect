import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
import numpy as np
import os
import pdb
import argparse
import logging
import time
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

parser = argparse.ArgumentParser(description="test_acc", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--num_batch", type=str)
args = parser.parse_args()

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# log = logging.getLogger("mylog")
# formatter = logging.Formatter("%(asctime)s : %(message)s")
# streamHandler = logging.StreamHandler()
# streamHandler.setFormatter(formatter)
# log.setLevel(logging.DEBUG)
# log.addHandler(streamHandler)

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.cpu().numpy()

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_ID = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

if args.dataset == "cifar10":
    ID_data = dset.CIFAR10('../data/cifar10', train=False, transform=transform_ID)
    if args.model == "wrn":
        net = WideResNet(40, 10, 2, dropRate = 0)
        net.load_state_dict(torch.load('./ckpt/cifar10_wrn_pretrained_epoch_99.pt'))
    elif args.model == "resnet18":
        net = models.resnet18()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet18_cifar10_epoch147.pt'))
    elif args.model == "resnet50":
        net = models.resnet50()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet50_cifar10_epoch124.pt'))
elif args.dataset == "cifar100":
    ID_data = dset.CIFAR100('../data/cifar100', train=False, transform=transform_ID)
    if args.model == "wrn":
        net = WideResNet(40, 100, 2, dropRate = 0)
        net.load_state_dict(torch.load('./ckpt/cifar100_wrn_pretrained_epoch_99.pt'))
    elif args.model == "resnet18":
        net = models.resnet18()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet18_cifar100_epoch109.pt'))
    elif args.model == "resnet50":
        net = models.resnet50()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet50_cifar100_epoch150.pt'))

OOD_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

ID_loader = torch.utils.data.DataLoader(
    ID_data, batch_size=128, shuffle=False
)

OOD_loader = torch.utils.data.DataLoader(
    OOD_data, batch_size=128, shuffle=False
)

OOD_num_examples = len(ID_data)

if torch.cuda.is_available():
    net.cuda()

def data_load():
    confident_scores_x = []
    confident_scores_y = []
    categories = []
    net.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ID_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            # pdb.set_trace()
            confident_scores_x.append(np.max(to_np(F.softmax(output, dim=1)), axis=1))
            confident_scores_y.append(torch.logsumexp(output.data.cpu(), dim=1).numpy())
            categories.append(np.where(to_np(pred.eq(target.data)), "ID_true_scores", "ID_false_scores"))
            if str(batch_idx + 1) == args.num_batch:
                break

        for batch_idx, (data, target) in enumerate(OOD_loader):
            # if batch_idx >= OOD_num_examples // 128:
            #     break
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            # pdb.set_trace()
            confident_scores_x.append(np.max(to_np(F.softmax(output, dim=1)), axis=1))
            confident_scores_y.append(torch.logsumexp(output.data.cpu(), dim=1).numpy())
            categories.append(np.where(to_np(pred.eq(target.data)), "OOD_scores", "OOD_scores"))
            if str(batch_idx + 1) == args.num_batch:
                return concat(confident_scores_x), concat(confident_scores_y), concat(categories)
        # return concat(confident_scores_x), concat(confident_scores_y), concat(categories)

confident_scores_x, confident_scores_y, categories = data_load()
# pdb.set_trace()
# print(confident_scores)
# print(categories)
scatter = sns.scatterplot(x=confident_scores_x, y=confident_scores_y, hue=categories, size=0.01)
scatter.get_figure().savefig("./" + args.model + "_" + args.num_batch + "_" + args.dataset + ".png")
# scatter.get_figure().savefig("./" + args.model + "_" + args.dataset + ".png")
