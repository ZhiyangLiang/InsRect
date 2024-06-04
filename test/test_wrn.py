import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset

import pdb
import argparse
import numpy as np
import models.resnet as resnet
from models.wrn import WideResNet

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = parser.parse_args()

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

id_data = dset.CIFAR10("../data/cifar10", train=False, transform=id_transform, download=False)
model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate = 0.3)
model.load_state_dict(torch.load('./ckpt/cifar10_wrn_epoch188_acc0.9493999481201172.pt'))

model = model.cuda()

def test(epoch):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    accuracy = correct / len(test_loader.dataset)
    log.debug("accuracy:" + str(accuracy))

for epoch in range(10):
    test(epoch)