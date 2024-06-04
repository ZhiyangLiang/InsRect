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
from ash import ash_p, ash_b, ash_s

parser = argparse.ArgumentParser(description="test_acc", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str)
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--score', type=str)
parser.add_argument('--use_ash', type=str)
parser.add_argument("--entropy_minimization_plus", type=str)
# parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
# parser.add_argument("--num_batch", type=str)
torch.cuda.set_device(1)
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

ID_transform = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.fc = nn.Linear(in_features=self.resnet18.fc.in_features, out_features=10, bias=True)
        # self.resnet18.fc = nn.Linear(in_features=self.resnet18.fc.in_features, out_features=100, bias=True)
        self.resnet18.load_state_dict(torch.load('./ckpt/resnet18_cifar10_epoch147.pt'))
        # self.resnet18.load_state_dict(torch.load('./ckpt/resnet18_cifar100_epoch160.pt'))
        self.conv1 = self.resnet18.conv1
        self.bn1 = self.resnet18.bn1
        self.relu = self.resnet18.relu
        self.maxpool = self.resnet18.maxpool
        self.layer1 = self.resnet18.layer1
        self.layer2 = self.resnet18.layer2
        self.layer3 = self.resnet18.layer3
        self.layer4 = self.resnet18.layer4
        self.avgpool = self.resnet18.avgpool
        self.fc = self.resnet18.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # x = ash_p(x)
        # x = ash_b(x)
        x = ash_s(x)

        x = self.fc(x.view(x.size(0), -1))

        # x = self.resnet18(x)
        return x

# newnet = ResNet18().cuda()

if args.dataset == "cifar10":
    ID_data = dset.CIFAR10('../data/cifar10', train=False, transform=ID_transform)
    if args.model == "wrn":
        net = WideResNet(40, 10, 2, dropRate = 0.3)
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
    ID_data = dset.CIFAR100('../data/cifar100', train=False, transform=ID_transform)
    if args.model == 'mobilenet':
        from models2.mobilenet import mobilenet
        net = mobilenet()
        if args.entropy_minimization_plus == "yes":
            net.load_state_dict(torch.load('./ckpt/mobilenet_epoch122_entropy_minimization_plus_acc0.6789000034332275.pt'))
        elif args.entropy_minimization_plus == "no":
            net.load_state_dict(torch.load('./ckpt/mobilenet_epoch124_acc0.677299976348877.pt'))
    elif args.model == 'resnet18':
        from models2.resnet import resnet18
        net = resnet18()
        if args.entropy_minimization_plus == "yes":
            net.load_state_dict(torch.load('./ckpt/resnet18_epoch197_entropy_minimization_plus_acc0.7547000050544739.pt'))
        elif args.entropy_minimization_plus == "no":
            net.load_state_dict(torch.load('./ckpt/resnet18_epoch190_acc0.7626000046730042.pt'))
    elif args.model == 'resnet50':
        from models2.resnet import resnet50
        net = resnet50()
        if args.entropy_minimization_plus == "yes":
            net.load_state_dict(torch.load('./ckpt/resnet50_epoch193_entropy_minimization_plus_acc0.7628999948501587.pt'))
        elif args.entropy_minimization_plus == "no":
            net.load_state_dict(torch.load('./ckpt/resnet50_epoch194_acc0.786799967288971.pt'))
    elif args.model == 'densenet161':
        from models2.densenet import densenet161
        net = densenet161()
        if args.entropy_minimization_plus == "yes":
            net.load_state_dict(torch.load('./ckpt/densenet161_epoch200_entropy_minimization_plus_acc0.7954999804496765.pt'))
        elif args.entropy_minimization_plus == "no":
            net.load_state_dict(torch.load('./ckpt/densenet161_epoch175_acc0.8011999726295471.pt'))
    elif args.model == 'wideresnet':
        from models2.wideresidual import wideresnet
        net = wideresnet()
        net.load_state_dict(torch.load('./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt'))

# OOD_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))
ID_loader = torch.utils.data.DataLoader(
    ID_data, batch_size=128, shuffle=False
)
# OOD_loader = torch.utils.data.DataLoader(
#     OOD_data, batch_size=128, shuffle=False
# )

if args.ood_dataset == "texture":
    texture_data = dset.ImageFolder(root="../data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "places365":
    places365_data = dset.ImageFolder(root="../data/places365", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "LSUN":
    lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "LSUN_resize":
    lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "iSUN":
    isun_data = dset.ImageFolder(root="../data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "LSUN_pil":
    lsun_pil_data = dset.ImageFolder(root="../data/LSUN_pil", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(lsun_pil_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "Imagenet_pil":
    imagenet_pil_data = dset.ImageFolder(root="../data/Imagenet_pil", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(imagenet_pil_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "Imagenet_resize":
    imagenet_resize_data = dset.ImageFolder(root="../data/Imagenet_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(imagenet_resize_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "102flowers":
    flowers_data = dset.ImageFolder(root="../data/102flowers", transform=trn.Compose([trn.Resize((32, 32)), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(flowers_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "Oxford_Pets":
    oxford_pets_data = dset.ImageFolder(root="../data/Oxford_Pets", transform=trn.Compose([trn.Resize((32, 32)), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(oxford_pets_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
elif args.ood_dataset == "Stanford_Dogs":
    stanford_dogs_data = dset.ImageFolder(root="../data/Stanford_Dogs", transform=trn.Compose([trn.Resize((32, 32)), trn.ToTensor(), trn.Normalize(mean, std)]))
    OOD_loader = torch.utils.data.DataLoader(stanford_dogs_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)

# OOD_num_examples = len(ID_data)

if torch.cuda.is_available():
    net.cuda()

def data_load():
    ID_true_scores = []
    ID_false_scores = []
    OOD_scores = []
    net.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ID_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = net(data)
            # output = newnet(data) # ash
            pred = output.data.max(1)[1]
            if args.score == "msp":
                ID_score = np.max(to_np(F.softmax(output, dim=1)), axis=1)
            elif args.score == "energy":
                ID_score = torch.logsumexp(output.data.cpu(), dim=1).numpy()
            ID_true_scores.append(ID_score[to_np(pred == target.data)])
            ID_false_scores.append(ID_score[to_np(pred != target.data)])
            # if str(batch_idx + 1) == args.num_batch:
            #     break

        for batch_idx, (data, target) in enumerate(OOD_loader):
            # if batch_idx >= OOD_num_examples // 128:
            #     break
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            if args.use_ash == "yes":
                output = newnet(data) # ash
            elif args.use_ash == "no":
                output = net(data)
            pred = output.data.max(1)[1]
            if args.score == "msp":
                OOD_score = np.max(to_np(F.softmax(output, dim=1)), axis=1)
            elif args.score == "energy":
                OOD_score = torch.logsumexp(output.data.cpu(), dim=1).numpy()
            OOD_scores.append(OOD_score)
            # if str(batch_idx + 1) == args.num_batch:
            #     return concat(ID_true_scores), concat(ID_false_scores), concat(OOD_scores)
        return concat(ID_true_scores), concat(ID_false_scores), concat(OOD_scores)

ID_true_scores, ID_false_scores, OOD_scores = data_load()
pdb.set_trace()

ax = sns.kdeplot(
    x=ID_true_scores,
    # bw_adjust=2,
    fill=True,
    alpha=0.5,
    linewidth=0,
    color='blue',
)
ax = sns.kdeplot(
    x=ID_false_scores,
    # bw_adjust=2,
    fill=True,
    alpha=0.5,
    linewidth=0,
    color='green'
)
ax = sns.kdeplot(
    x=OOD_scores,
    # bw_adjust=2,
    fill=True,
    alpha=0.5,
    linewidth=0,
    color='yellow'
)
ax.legend(labels=["y=ID_true", "y=ID_false", "y=OOD"])
if args.use_ash == "yes":
    if args.entropy_minimization_plus == "yes":
        ax.set(title=args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + "(combined with ash)_entropy_minimization_plus") plt.savefig("./" + args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + "(combined with ash)_entropy_minimization_plus.png")
    elif args.entropy_minimization_plus == "no":
        ax.set(title=args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + "(combined with ash)")
        plt.savefig("./" + args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + "(combined with ash).png")
elif args.use_ash == "no":
    if args.entropy_minimization_plus == "yes":
        ax.set(title=args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + "_entropy_minimization_plus")
        plt.savefig("./" + args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + "_entropy_minimization_plus.png")
    elif args.entropy_minimization_plus == "no":
        ax.set(title=args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score)
        plt.savefig("./" + args.model + "_" + args.dataset + "_" + args.ood_dataset + "_" + args.score + ".png")
