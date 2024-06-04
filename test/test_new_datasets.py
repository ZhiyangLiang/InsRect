import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset

import pdb
import argparse
import logging
import time
import torch.optim as optim

import models.hybrid_resnet as resnet
from models.densenet import densenet161
from models.resnet_cifar_ash import ResNet34, ResNet50
from models.densenet_dice import DenseNet3
# from models.densenet_ash import DenseNet3
from utils.svhn_loader import SVHN
import numpy as np
import sklearn.metrics as sk
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import NMF, FastICA
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from utils.tinyimages_300K_random_loader import TinyImages

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()
id_feats = []
id_feats_test = []
id_feats_eval = []
ood_feats = []
ood1_feats = []
ood2_feats = []
ood3_feats = []
ood4_feats = []
ood5_feats = []
ood6_feats = []
ood7_feats = []
ood8_feats = []
ood9_feats = []
ood10_feats = []
ood11_feats = []
ood12_feats = []

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

log = logging.getLogger("InsRect")

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])


id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)

model = DenseNet3(100, 100)
model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
model = model.cuda()

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

# ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
# ood1_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
# ood2_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
# ood3_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
# ood4_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
# ood5_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
# ood6_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)
ood7_data = dset.ImageFolder(root="../data/LSUN_pil", transform=eval_transform)
ood8_data = dset.ImageFolder(root="../data/Imagenet_resize", transform=eval_transform)
ood9_data = dset.ImageFolder(root="../data/Imagenet_pil", transform=eval_transform)
ood10_data = dset.ImageFolder(root="../data/CUB_200_2011/images", transform=eval_transform)
ood11_data = dset.ImageFolder(root="../data/Stanford_Dogs",transform=eval_transform)
ood12_data = dset.ImageFolder(root="../data/Oxford_Pets",transform=eval_transform)

# id_loader = torch.utils.data.DataLoader(id_data, batch_size=200, shuffle=True, num_workers=4)
# id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=200, shuffle=True, num_workers=4)
# id_loader_eval = torch.utils.data.DataLoader(id_data_test, batch_size=200, shuffle=True, num_workers=4)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=200, shuffle=True, num_workers=4)
# ood1_loader = torch.utils.data.DataLoader(ood1_data, batch_size=200, shuffle=True, num_workers=4)
# ood2_loader = torch.utils.data.DataLoader(ood2_data, batch_size=200, shuffle=True, num_workers=4)
# ood3_loader = torch.utils.data.DataLoader(ood3_data, batch_size=200, shuffle=True, num_workers=4)
# ood4_loader = torch.utils.data.DataLoader(ood4_data, batch_size=200, shuffle=True, num_workers=4)
# ood5_loader = torch.utils.data.DataLoader(ood5_data, batch_size=200, shuffle=True, num_workers=4)
# ood6_loader = torch.utils.data.DataLoader(ood6_data, batch_size=200, shuffle=True, num_workers=4)
ood7_loader = torch.utils.data.DataLoader(ood7_data, batch_size=200, shuffle=True, num_workers=4)
ood8_loader = torch.utils.data.DataLoader(ood8_data, batch_size=200, shuffle=True, num_workers=4)
ood9_loader = torch.utils.data.DataLoader(ood9_data, batch_size=200, shuffle=True, num_workers=4)
ood10_loader = torch.utils.data.DataLoader(ood10_data, batch_size=200, shuffle=True, num_workers=4)
ood11_loader = torch.utils.data.DataLoader(ood11_data, batch_size=200, shuffle=True, num_workers=4)
ood12_loader = torch.utils.data.DataLoader(ood12_data, batch_size=200, shuffle=True, num_workers=4)

def extract_feats(feats, loader, opt=0):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            if opt == 1:
                data = data + torch.empty_like(data).normal_(0, 0.005) # Gaussian Noise
            feats.append(model.get_features_fc(data))

# extract_feats(ood1_feats, ood1_loader)
# extract_feats(ood2_feats, ood2_loader)
# extract_feats(ood3_feats, ood3_loader)
# extract_feats(ood4_feats, ood4_loader)
# extract_feats(ood5_feats, ood5_loader)
# extract_feats(ood6_feats, ood6_loader)
extract_feats(ood7_feats, ood7_loader)
extract_feats(ood8_feats, ood8_loader)
extract_feats(ood9_feats, ood9_loader)
extract_feats(ood10_feats, ood10_loader)
extract_feats(ood11_feats, ood11_loader)
extract_feats(ood12_feats, ood12_loader)

ood7_feats = torch.cat(ood7_feats, dim=0)
ood8_feats = torch.cat(ood8_feats, dim=0)
ood9_feats = torch.cat(ood9_feats, dim=0)
ood10_feats = torch.cat(ood10_feats, dim=0)
ood11_feats = torch.cat(ood11_feats, dim=0)
ood12_feats = torch.cat(ood12_feats, dim=0)

pdb.set_trace()
print(1)
