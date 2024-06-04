import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import numpy as np
import os
import pdb
import argparse
import logging
import time
import pdb
import resnet

parser = argparse.ArgumentParser(description="test_acc", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--model", type=str)
# parser.add_argument("--dataset", type=str)
args = parser.parse_args()

torch.cuda.set_device(0)
torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

log = logging.getLogger("mylog")
formatter = logging.Formatter("%(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(streamHandler)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_test = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

net = resnet.resnet50(num_classes=100)
net.load_state_dict(torch.load("./resnet50_cifar100_0.7828.pt"))
net.cuda()

test_data = dset.CIFAR100('../data/cifar100', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=128, shuffle=False
)

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
