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

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "wrn"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)

parser.add_argument('--epochs', '-e', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--select', type=int, default=50)
parser.add_argument('--sample_from', type=int, default=10000)
args = parser.parse_args()

start_epoch = 0
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

vos_feats = []

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
    num_classes = 10
    if args.model == "resnet50":
        net = resnet.resnet50(num_classes=10)
        net.load_state_dict(torch.load("./ckpt/resnet50_cifar10-192-best-0.9546999931335449.pth"))
    elif args.model == "densenet_dice":
        net = DenseNet3(100, 10)
        net.load_state_dict(torch.load("./ckpt/checkpoint_10.pth.tar")["state_dict"])
        data_dict = torch.zeros(num_classes, args.sample_number, 342).cuda()
        eye_matrix = torch.eye(342, device='cuda')
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
    num_classes = 100
    if args.model == "resnet50":
        net = resnet.resnet50(num_classes=100)
        net.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
    elif args.model == "densenet_dice":
        net = DenseNet3(100, 100)
        net.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
        data_dict = torch.zeros(num_classes, args.sample_number, 342).cuda()
        eye_matrix = torch.eye(342, device='cuda')
    elif args.model == "wideresnet":
        net = wideresnet(num_classes=100)
        net.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
    elif args.model == "mobilenet":
        net = mobilenet(class_num=100)
        net.load_state_dict(torch.load("./ckpt/mobilenet_epoch124_acc0.677299976348877.pt", map_location='cuda:0'))
    elif args.model == "wrn":
        net = WideResNet(40, 100, 2, dropRate=0.3)
        net.load_state_dict(torch.load("./ckpt/cifar100_wrn_pretrained_epoch_99.pt"))

number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
weight_energy = torch.nn.Linear(num_classes, 1).cuda()
torch.nn.init.uniform_(weight_energy.weight)

net = net.cuda()
ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)

train_loader = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
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

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    import math
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)

def train(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0
    print("epoch: %d" % (epoch))
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)


        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                      output[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                      output[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((args.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, args.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)

            vos_feats.append(ood_samples)
            if idx == 19:
                vos_feats_save = torch.cat(vos_feats, dim=0)
                print(vos_feats_save.shape)
                torch.save(vos_feats_save, "../data/vos_cifar100_densenet_dice_19.pkl")
            elif idx == 24:
                vos_feats_save = torch.cat(vos_feats, dim=0)
                print(vos_feats_save.shape)
                torch.save(vos_feats_save, "../data/vos_cifar100_densenet_dice_24.pkl")
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

for epoch in range(start_epoch, args.epochs):
    train(epoch)
