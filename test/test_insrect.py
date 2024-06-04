import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset

import models.hybrid_resnet as resnet
from models.densenet import densenet161
from models.densenet_dice import DenseNet3

import pdb
import argparse
import logging
import time
import torch.optim as optim
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet161", "densenet_dice", "densenet_ash"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train_trans", type=str, choices=["id", "train"])
parser.add_argument("--auxiliary_trans", type=str, choices=["ood", "eval"])
parser.add_argument("--width", type=float)
parser.add_argument("--start", type=int)
args = parser.parse_args()

recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()
id_feats_invariant = []
ood_feats_invariant = []
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if args.model == "resnet50":
    model = resnet.resnet50(num_classes=100)
    model.load_state_dict(torch.load("./ckpt/resnet50_cifar100_0.7828.pt"))
elif args.model == "densenet161":
    model = densenet161()
    model.load_state_dict(torch.load("./ckpt/densenet161_epoch175_acc0.8011999726295471.pt"))
elif args.model == "densenet_dice":
    model = DenseNet3(100, 100)
    model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
elif args.model == "densenet_ash":
    model = DenseNet3(100, 100)
    model.load_state_dict(torch.load("./ckpt/densenet161_ash_cifar100_epoch136_acc0.7649999856948853.pt"))
model = model.cuda()

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

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

if args.train_trans == "train":
    id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
elif args.train_trans == "id":
    id_data = dset.CIFAR100("../data/cifar100", train=True, transform=id_transform, download=False)
id_loader = torch.utils.data.DataLoader(id_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# if args.auxiliary_trans == "ood":
#     ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
# elif args.auxiliary_trans == "eval":
#     ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=eval_transform)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

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
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _score.append(-to_np(torch.logsumexp(output, axis=1)))
    return concat(_score).copy()

def get_and_print_results(mylog, ood_loader, in_score):
    model.eval()
    aurocs, auprs, fprs = [], [], []
    ood_score = get_ood_scores(ood_loader)
    measures = get_measures(ood_score, in_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(mylog, auroc, aupr, fpr)
    return fpr, auroc, aupr

def score_get_and_print_results(mylog, in_score, ood_score):
    model.eval()
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

def test(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(loader.dataset) * 100

def extract_feats(feats, loader, opt=0):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            if opt == 1:
                data = data + torch.empty_like(data).normal_(0, 0.005) # Gaussian Noise
            feats.append(model.get_features_fc(data))

extract_feats(id_feats_invariant, id_loader)
id_feats_invariant = torch.cat(id_feats_invariant, dim=0)
# extract_feats(ood_feats_invariant, ood_loader)
# ood_feats_invariant = torch.cat(ood_feats_invariant, dim=0)[:50000]
nmf = NMF(n_components=3, max_iter=20000)
nmf.fit(id_feats_invariant.cpu())

# plt.bar(range(342), nmf.components_[0], color=['cyan'])
# plt.ylim((0.0, 7.5))
# # plt.ylim((0.0, 16))
# plt.xticks([])
# plt.yticks([])
# plt.savefig("./component1_cyan_7.5.svg")
# # plt.savefig("./component1_cyan_16.png")
# plt.clf()
# plt.bar(range(342), nmf.components_[1], color=['lime'])
# plt.ylim((0.0, 7.5))
# # plt.ylim((0.0, 16))
# plt.xticks([])
# plt.yticks([])
# plt.savefig("./component2_lime_7.5.svg")
# # plt.savefig("./component2_lime_16.png")
# plt.clf()
# plt.bar(range(342), nmf.components_[2], color=['red'])
# plt.ylim((0.0, 7.5))
# # plt.ylim((0.0, 16))
# plt.xticks([])
# plt.yticks([])
# plt.savefig("./component3_red_7.5.svg")
# # plt.savefig("./component3_red_16.png")
# plt.clf()

def my_plt_comp(comp_num):
    nmf_comp = torch.clone(torch.Tensor(nmf.components_[comp_num]))
    nmf_comp[nmf_comp > 7.5] = 7.5
    if comp_num == 0:
        plt.bar(range(342 - args.start), nmf_comp[args.start:], color=['paleturquoise'], width=args.width)
    if comp_num == 1:
        plt.bar(range(342 - args.start), nmf_comp[args.start:], color=['palegreen'], width=args.width)
    if comp_num == 2:
        plt.bar(range(342 - args.start), nmf_comp[args.start:], color=['lightcoral'], width=args.width)
    plt.ylim((0.0, 7.5))
    plt.xticks([])
    plt.yticks([])

    if comp_num == 0:
        nmf_comp[nmf_comp < 1.053] = 0
        nmf_comp[nmf_comp > 1.697] = 1.697
        plt.bar(range(342 - args.start), nmf_comp[args.start:], color=['cyan'], width=args.width)
    if comp_num == 1:
        nmf_comp[nmf_comp < 1.032] = 0
        nmf_comp[nmf_comp > 1.755] = 1.755
        plt.bar(range(342 - args.start), nmf_comp[args.start:], color=['lime'], width=args.width)
    if comp_num == 2:
        nmf_comp[nmf_comp < 0] = 0
        nmf_comp[nmf_comp > 1.738] = 1.738
        plt.bar(range(342 - args.start), nmf_comp[args.start:], color=['red'], width=args.width)
    plt.ylim((0.0, 7.5))
    plt.xticks([])
    plt.yticks([])
    if comp_num == 0:
        # plt.savefig("./component0_cyan_new.png")
        plt.savefig("./component0_cyan_new.svg")
    if comp_num == 1:
        # plt.savefig("./component1_lime_new.png")
        plt.savefig("./component1_lime_new.svg")
    if comp_num == 2:
        # plt.savefig("./component2_red_new.png")
        plt.savefig("./component2_red_new.svg")
    plt.clf()

my_plt_comp(0)
my_plt_comp(1)
my_plt_comp(2)
