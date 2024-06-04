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

import models.resnet_train as resnet
from models.densenet_dice import DenseNet3
from models.wideresidual import wideresnet
from models.mobilenet import mobilenet
from utils.svhn_loader import SVHN
import numpy as np
import sklearn.metrics as sk
from sklearn.decomposition import PCA, NMF, FastICA
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from utils.tinyimages_300K_random_loader import TinyImages
from models.wrn import WideResNet
import joblib
import sys
import utils.utils_awp as awp

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "wrn"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)

parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--begin_epoch', type=int, default=0)
args = parser.parse_args()

recall_level_default = 0.95
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
texture_feats = []
places365_feats = []
lsunc_feats = []
lsunr_feats = []
isun_feats = []
svhn_feats = []

doe_feats = []

torch.cuda.set_device(0)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

log = logging.getLogger("InsRect")
nmf_relu = nn.ReLU(inplace=True)

if 'cifar' in args.dataset:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
else:
    mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])

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

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('../data/cifar10', train=True, transform=train_transform)
    test_data = dset.CIFAR10('../data/cifar10', train=False, transform=id_transform)

    if args.model == "resnet50":
        net = resnet.resnet50(num_classes=10)
        net.load_state_dict(torch.load("./ckpt/resnet50_cifar10-192-best-0.9546999931335449.pth"))
    elif args.model == "densenet_dice":
        net = DenseNet3(100, 10)
        net.load_state_dict(torch.load("./ckpt/checkpoint_10.pth.tar")["state_dict"])
    elif args.model == "wideresnet":
        net = wideresnet(num_classes=10)
        net.load_state_dict(torch.load("./ckpt/wideresnet_cifar10_epoch195_acc0.960599958896637.pt", map_location='cuda:0'))
    elif args.model == "mobilenet":
        net = mobilenet(class_num=10)
        net.load_state_dict(torch.load("./ckpt/mobilenet_cifar10_epoch183_acc0.90829998254776.pt", map_location='cuda:0'))
    elif args.model == "wrn":
        net = WideResNet(40, 10, 2, dropRate=0.3)
        net.load_state_dict(torch.load("./ckpt/cifar10_wrn_pretrained_epoch_99.pt"))
elif args.dataset == 'cifar100':
    train_data_in = dset.CIFAR100('../data/cifar100', train=True, transform=train_transform)
    test_data = dset.CIFAR100('../data/cifar100', train=False, transform=id_transform)

    if args.model == "resnet50":
        net = resnet.resnet50(num_classes=100)
        net.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
    elif args.model == "densenet_dice":
        net = DenseNet3(100, 100)
        net.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
    elif args.model == "wideresnet":
        net = wideresnet(num_classes=100)
        net.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
    elif args.model == "mobilenet":
        net = mobilenet(class_num=100)
        net.load_state_dict(torch.load("./ckpt/mobilenet_epoch124_acc0.677299976348877.pt", map_location='cuda:0'))
    elif args.model == "wrn":
        net = WideResNet(40, 100, 2, dropRate=0.3)
        net.load_state_dict(torch.load("./ckpt/cifar100_wrn_pretrained_epoch_99.pt"))

net = net.cuda()
ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)

train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    examples[np.isnan(examples)] = 0.0
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def print_measures(mylog, auroc, aupr, fpr):
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))
    mylog.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))

def get_ood_scores(loader):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            _score.append(-to_np(torch.logsumexp(output, axis=1)))
    return concat(_score).copy()

def get_and_print_results(mylog, ood_loader, in_score):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    ood_score = get_ood_scores(ood_loader)
    measures = get_measures(ood_score, in_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(mylog, auroc, aupr, fpr)
    return fpr, auroc, aupr

def score_get_and_print_results(mylog, in_score, ood_score):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(ood_score, in_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(mylog, auroc, aupr, fpr)
    return fpr, auroc, aupr

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64) # 累积加法, 即最后一个得到的元素为数组中全部元素之和
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    # return fps[cutoff] / (np.sum(np.logical_not(y_true))) # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
    return (fps[cutoff] / (np.sum(np.logical_not(y_true))))

def train(epoch, diff):
    if args.dataset == "cifar10":
        if args.model == "resnet50":
            proxy = resnet.resnet50(num_classes=10)
        elif args.model == "densenet_dice":
            proxy = DenseNet3(100, 10)
        elif args.model == "wideresnet":
            proxy = wideresnet(num_classes=10)
        elif args.model == "mobilenet":
            proxy = mobilenet(class_num=10)
        elif args.model == "wrn":
            proxy = WideResNet(40, 10, 2, dropRate=0.3)
    elif args.dataset == "cifar100":
        if args.model == "resnet50":
            proxy = resnet.resnet50(num_classes=100)
        elif args.model == "densenet_dice":
            proxy = DenseNet3(100, 100)
        elif args.model == "wideresnet":
            proxy = wideresnet(num_classes=100)
        elif args.model == "mobilenet":
            proxy = mobilenet(class_num=100)
        elif args.model == "wrn":
            proxy = WideResNet(40, 100, 2, dropRate=0.3)
    proxy = proxy.cuda()

    proxy_optim = torch.optim.SGD(proxy.parameters(), lr=1)

    net.train()

    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        if epoch >= args.warmup:
            if args.dataset == 'cifar10':
                gamma =  torch.Tensor([1e-1,1e-2,1e-3,1e-4])[torch.randperm(4)][0]
            else: 
                gamma =  torch.Tensor([1e-1,1e-2,1e-3,1e-4])[torch.randperm(4)][0] # 31
            proxy.load_state_dict(net.state_dict())
            proxy.train()
            scale = torch.Tensor([1]).cuda().requires_grad_()
            feats = proxy.get_features_fc(data)
            x = proxy.fc(feats) * scale
            # x = proxy(data) * scale
            l_sur = (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            # l_sur = - (x.log_softmax(1) * (x / 0.1).softmax(1).detach()).sum(-1).mean()
            reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph = True)[0] ** 2)
            proxy_optim.zero_grad()
            reg_sur.backward()
            # l_sur.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            proxy_optim.step()
            if epoch == args.warmup and batch_idx == 0:
                diff = awp.diff_in_weights(net, proxy)
            else:
                # diff = awp.diff_in_weights(net, proxy)
                diff = awp.average_diff(diff, awp.diff_in_weights(net, proxy), beta = .6)

            awp.add_into_weights(net, diff, coeff = gamma)
        
        feats = net.get_features_fc(data)
        x = net.fc(feats)
        # x = net(data)
        if epoch >= args.warmup:
            doe_feats.append(feats)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        l_oe = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
        if args.dataset == 'cifar10':
            if epoch >= args.warmup:
                loss = l_oe
            else:
                loss = l_ce +  l_oe
        else: 
            if epoch >= args.warmup:
                loss = l_oe
            else: 
                loss = l_ce +  l_oe
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1) 
        optimizer.step()

        if epoch >= args.warmup:
            awp.add_into_weights(net, diff, coeff = - gamma)
            optimizer.zero_grad()
            feats = net.get_features_fc(data)
            x = net.fc(feats)
            # x = net(data)
            l_ce = F.cross_entropy(x[:len(in_set[0])], target)
            loss = l_ce # + l_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %(epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        scheduler.step()

    if epoch >= args.warmup:
        torch.save(torch.cat(doe_feats, dim=0), "doe_cifar100_densenet_dice_%d.pkl" % (epoch))
    print()
    return diff

def test(loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(loader.dataset) * 100

optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))
diff = None

print('Beginning Training\n')
for epoch in range(args.begin_epoch, args.epochs - 1):
    diff = train(epoch, diff)
