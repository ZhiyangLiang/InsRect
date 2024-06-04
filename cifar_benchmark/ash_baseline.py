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
# from models.densenet import densenet161
# from models.resnet_cifar_ash import ResNet34, ResNet50
from models.densenet_dice import DenseNet3
from models.wideresidual import wideresnet
from models.mobilenet import mobilenet
from utils.svhn_loader import SVHN
import numpy as np
import sklearn.metrics as sk
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import NMF, FastICA
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from utils.tinyimages_300K_random_loader import TinyImages
from sklearn.decomposition import NMF

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet18", "resnet34", "resnet50", "densenet_dice", "wideresnet", "mobilenet"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--per_ash", type=int)
args = parser.parse_args()

final_fpr = 1.0
final_avg_fpr = 1.0
recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()
id_feats = []
id_feats_test = []
ood_feats = []
texture_feats = []
places365_feats = []
lsunc_feats = []
lsunr_feats = []
isun_feats = []
svhn_feats = []

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

log = logging.getLogger("InsRect")

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

def ash_s(x, percentile): # nmf_relu_scale_new_version
    x_relu = nmf_relu(x)
    s1 = x.sum(dim=1)
    n = x.shape[1]
    k = int(n * percentile / 100)
    # k = math.ceil(n * percentile / 100)
    t = x
    v, i = torch.topk(t, k, dim=1)
    s2 = v.sum(dim=1)
    scale = s1 / s2
    t.scatter_(dim=1, index=i, src=v * torch.exp(scale[:, None]))
    return x

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])

eval_transform = trn.Compose([
    trn.Resize(32),
    trn.CenterCrop(32),
    trn.ToTensor(),
    trn.Normalize(mean, std)
])

if args.dataset == "cifar10":
    id_data_test = dset.CIFAR10("../data/cifar10", train=False, transform=id_transform, download=False)
    if args.model == "resnet18":
        model = resnet.resnet18(num_classes=10, per_ash=args.per_ash)
        model.load_state_dict(torch.load("./ckpt/resnet18_cifar10-183-best-0.9485999941825867.pth"))
    elif args.model == "resnet34":
        model = resnet.resnet34(num_classes=10, per_ash=args.per_ash)
        model.load_state_dict(torch.load("./ckpt/resnet34_cifar10-185-best-0.9529999494552612.pth"))
    elif args.model == "resnet50":
        model = resnet.resnet50(num_classes=10, per_ash=args.per_ash)
        model.load_state_dict(torch.load("./ckpt/resnet50_cifar10-192-best-0.9546999931335449.pth"))
    elif args.model == "densenet_dice":
        model = DenseNet3(100, 10)
        model.load_state_dict(torch.load("./ckpt/checkpoint_10.pth.tar")["state_dict"])
    elif args.model == "wideresnet":
        model = wideresnet(num_classes=10)
        model.load_state_dict(torch.load("./ckpt/wideresnet_cifar10_epoch195_acc0.960599958896637.pt", map_location='cuda:0'))
    elif args.model == "mobilenet":
        model = mobilenet(class_num=10)
        model.load_state_dict(torch.load("./ckpt/mobilenet_cifar10_epoch183_acc0.90829998254776.pt", map_location='cuda:0'))
    
elif args.dataset == "cifar100":
    id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)
    if args.model == "resnet18":
        model = resnet.resnet18(num_classes=100, per_ash=args.per_ash)
        model.load_state_dict(torch.load("./ckpt/resnet18_cifar100-198-best-0.7594000101089478.pth"))
    elif args.model == "resnet34":
        model = resnet.resnet34(num_classes=100, per_ash=args.per_ash)
        model.load_state_dict(torch.load("./ckpt/resnet34_cifar100-192-best-0.774899959564209.pth"))
    elif args.model == "resnet50":
        model = resnet.resnet50(num_classes=100, per_ash=args.per_ash)
        model.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
    elif args.model == "densenet_dice":
        model = DenseNet3(100, 100)
        model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
    elif args.model == "wideresnet":
        model = wideresnet(num_classes=100)
        model.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
    elif args.model == "mobilenet":
        model = mobilenet(class_num=100)
        model.load_state_dict(torch.load("./ckpt/mobilenet_epoch124_acc0.677299976348877.pt", map_location='cuda:0'))

model = model.cuda()

texture_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
places365_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
isun_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
svhn_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

# nmf = NMF(n_components=args.num_component, max_iter=20000)
# nmf.fit(id_feats.cpu())

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

def react_percent(x, percent):
    threshold = np.percentile(x, percent)
    x = torch.clip(x, max=threshold)
    return x

def ash_s_thre(x, threshold):
    s1 = x.sum()
    k = (x >= threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum()
    scale = s1 / s2
    x *= torch.exp(scale)
    return x

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
            feats.append(model(data))

acc = test(id_loader_test)
print("acc: %.4f" % (acc))

extract_feats(id_feats_test, id_loader_test)
# extract_feats(texture_feats, texture_loader)
# extract_feats(places365_feats, places365_loader)
# extract_feats(lsunc_feats, lsunc_loader)
# extract_feats(lsunr_feats, lsunr_loader)
# extract_feats(isun_feats, isun_loader)
# extract_feats(svhn_feats, svhn_loader)

id_logits_test = torch.cat(id_feats_test, dim=0)
texture_logits= torch.cat(texture_feats, dim=0)
places365_logits = torch.cat(places365_feats, dim=0)
lsunc_logits = torch.cat(lsunc_feats, dim=0)
lsunr_logits = torch.cat(lsunr_feats, dim=0)
isun_logits = torch.cat(isun_feats, dim=0)
svhn_logits = torch.cat(svhn_feats, dim=0)

id_score_test =  - torch.logsumexp(id_logits_test, axis=1).cpu().detach().numpy()
texture_score =  - torch.logsumexp(texture_logits, axis=1).cpu().detach().numpy()
places365_score =  - torch.logsumexp(places365_logits, axis=1).cpu().detach().numpy()
lsunc_score =  - torch.logsumexp(lsunc_logits, axis=1).cpu().detach().numpy()
lsunr_score =  - torch.logsumexp(lsunr_logits, axis=1).cpu().detach().numpy()
isun_score =  - torch.logsumexp(isun_logits, axis=1).cpu().detach().numpy()
svhn_score =  - torch.logsumexp(svhn_logits, axis=1).cpu().detach().numpy()

def evaluate():
    texture_fpr, texture_auroc, _ = score_get_and_print_results(log, id_score_test, texture_score)
    places365_fpr, places365_auroc, _ = score_get_and_print_results(log, id_score_test, places365_score)
    lsunc_fpr, lsunc_auroc, _ = score_get_and_print_results(log, id_score_test, lsunc_score)
    lsunr_fpr, lsunr_auroc, _ = score_get_and_print_results(log, id_score_test, lsunr_score)
    isun_fpr, isun_auroc, _ = score_get_and_print_results(log, id_score_test, isun_score)
    svhn_fpr, svhn_auroc, _ = score_get_and_print_results(log, id_score_test, svhn_score)
    print("avg_fpr: %.2f" % ((texture_fpr + places365_fpr + lsunc_fpr + lsunr_fpr + isun_fpr + svhn_fpr) / 6 * 100))
    print("avg_auroc: %.2f" % ((texture_auroc + places365_auroc + lsunc_auroc + lsunr_auroc + isun_auroc + svhn_auroc) / 6 * 100))
evaluate()
