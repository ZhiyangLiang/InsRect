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

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet161", "densenet_dice", "densenet_ash"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train_trans", type=str, choices=["id", "train"])
parser.add_argument("--auxiliary_trans", type=str, choices=["ood", "eval"])
parser.add_argument("--idtype", type=str, choices=["yes", "no"])
parser.add_argument("--oodtype", type=str, choices=["yes", "no"])
parser.add_argument("--num_comp", type=int)
parser.add_argument("--width", type=float)
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

if args.idtype == "yes":
    if args.train_trans == "train":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
    elif args.train_trans == "id":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=id_transform, download=False)
    id_loader = torch.utils.data.DataLoader(id_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
if args.oodtype == "yes":
    if args.auxiliary_trans == "ood":
        ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
    elif args.auxiliary_trans == "eval":
        ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=eval_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

if args.idtype == "yes":
    extract_feats(id_feats_invariant, id_loader)
    id_feats_invariant = torch.cat(id_feats_invariant, dim=0)
if args.oodtype == "yes":
    extract_feats(ood_feats_invariant, ood_loader)
    ood_feats_invariant = torch.cat(ood_feats_invariant, dim=0)[:50000]

def my_plt_bar(item_num, num_comp, pictype, datatype, percent):
    if datatype == "id":
        global id_feats_invariant
        id_feats = torch.clone(id_feats_invariant)
        if pictype == "origin":
            plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['cornflowerblue'], width=args.width)
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['dodgerblue'], width=args.width)
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['blue'], width=args.width)
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_cornflowerblue.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_cornflowerblue.svg")

            plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['royalblue'], width=args.width)
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_royalblue.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_royalblue.svg")
        elif pictype == "ash":
            thre = np.percentile(id_feats.cpu(), percent)
            id_feats[id_feats >= thre] = 0
            plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['lightsteelblue'], width=args.width)
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['lightskyblue'])
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['cyan'])
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(percent) + "_" + str(args.width) + "_lightsteelblue.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_lightsteelblue_" + str(percent) + "_" + str(args.width) + ".svg")
            plt.clf()
        elif pictype == "react1":
            plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['lightsteelblue'], width=args.width)
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['lightskyblue'])
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['cyan'])
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_lightsteelblue.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_lightsteelblue.svg")
        elif pictype == "react2":
            id_feats[id_feats > 1] = 1
            plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['cornflowerblue'], width=args.width)
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['dodgerblue'])
            # plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['blue'])
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_cornflowerblue.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_cornflowerblue.svg")

            plt.bar(range(num_comp), id_feats[item_num][:num_comp].cpu(), color=['royalblue'], width=args.width)
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_royalblue.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_royalblue.svg")
            plt.clf()
        plt.ylim((0.0, 7.5))
    elif datatype == "ood":
        global ood_feats_invariant
        ood_feats = torch.clone(ood_feats_invariant)
        if pictype == "origin":
            plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['dimgrey'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['red'])
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_dimgrey.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_dimgrey.svg")
        elif pictype == "ash":
            thre = np.percentile(ood_feats.cpu(), percent)
            ood_feats[ood_feats >= thre] = 0
            plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['silver'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['lightcoral'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['palevioletred'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['violet'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['orchid'], width=args.width)
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(percent) + "_" + str(args.width) + "_silver.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(percent) + "_" + str(args.width) + "_silver.svg")

            plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['lightgrey'], width=args.width)
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(percent) + "_" + str(args.width) + "_lightgrey.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(percent) + "_" + str(args.width) + "_lightgrey.svg")
            plt.clf()
        elif pictype == "react1":
            plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['silver'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['lightcoral'])if args.idtype == "yes":
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['palevioletred'])
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['violet'])
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['orchid'])
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_silver.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_silver.svg")

            plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['lightgrey'], width=args.width)
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_lightgrey.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_lightgrey.svg")
        elif pictype == "react2":
            ood_feats[ood_feats > 1] = 1
            plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['dimgrey'], width=args.width)
            # plt.bar(range(num_comp), ood_feats[item_num][:num_comp].cpu(), color=['red'])
            plt.ylim((0.0, 7.5))
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_dimgrey.png")
            # plt.savefig("./" + datatype + "_" + pictype + "_" + str(item_num) + "_" + str(args.width) + "_dimgrey.svg")
            plt.clf()

# choose datapoints
# for i in range(20):
#     if args.idtype == "yes":
#         my_plt_bar(i, 0, "origin", "id")
#         # plt.clf()
#         my_plt_bar(i, 0, "ash", "id")
#         my_plt_bar(i, 0, "react1", "id")
#         my_plt_bar(i, 0, "react2", "id")
#     if args.oodtype == "yes":
#         my_plt_bar(i, 0, "origin", "ood")
#         # plt.clf()
#         my_plt_bar(i, 0, "ash", "ood")
#         my_plt_bar(i, 0, "react1", "ood")
#         my_plt_bar(i, 0, "react2", "ood")

# draw
if args.idtype == "yes":
    my_plt_bar(7, args.num_comp, "origin", "id", 90)
    my_plt_bar(7, args.num_comp, "ash", "id", 90)
    my_plt_bar(7, args.num_comp, "react1", "id", 90)
    my_plt_bar(7, args.num_comp, "react2", "id", 90)
    my_plt_bar(7, args.num_comp, "origin", "id", 80)
    my_plt_bar(7, args.num_comp, "ash", "id", 80)
    my_plt_bar(7, args.num_comp, "react1", "id", 80)
    my_plt_bar(7, args.num_comp, "react2", "id", 80)
    my_plt_bar(7, args.num_comp, "origin", "id", 70)
    my_plt_bar(7, args.num_comp, "ash", "id", 70)
    my_plt_bar(7, args.num_comp, "react1", "id", 70)
    my_plt_bar(7, args.num_comp, "react2", "id", 70)
    my_plt_bar(7, args.num_comp, "origin", "id", 60)
    my_plt_bar(7, args.num_comp, "ash", "id", 60)
    my_plt_bar(7, args.num_comp, "react1", "id", 60)
    my_plt_bar(7, args.num_comp, "react2", "id", 60)
if args.oodtype == "yes":
    my_plt_bar(3, args.num_comp, "origin", "ood", 90)
    my_plt_bar(3, args.num_comp, "ash", "ood", 90)
    my_plt_bar(3, args.num_comp, "react1", "ood", 90)
    my_plt_bar(3, args.num_comp, "react2", "ood", 90)
    my_plt_bar(3, args.num_comp, "origin", "ood", 80)
    my_plt_bar(3, args.num_comp, "ash", "ood", 80)
    my_plt_bar(3, args.num_comp, "react1", "ood", 80)
    my_plt_bar(3, args.num_comp, "react2", "ood", 80)
    my_plt_bar(3, args.num_comp, "origin", "ood", 70)
    my_plt_bar(3, args.num_comp, "ash", "ood", 70)
    my_plt_bar(3, args.num_comp, "react1", "ood", 70)
    my_plt_bar(3, args.num_comp, "react2", "ood", 70)
    my_plt_bar(3, args.num_comp, "origin", "ood", 60)
    my_plt_bar(3, args.num_comp, "ash", "ood", 60)
    my_plt_bar(3, args.num_comp, "react1", "ood", 60)
    my_plt_bar(3, args.num_comp, "react2", "ood", 60)


