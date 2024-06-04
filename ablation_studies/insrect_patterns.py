import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset
import torchvision.models as models
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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "wrn"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--method", type=str, choices=["pca", "nmf", "ica"])
parser.add_argument("--iter", type=int, choices=[0, 10, 20, 50, 100, 200, 500])
parser.add_argument("--index", type=int)
args = parser.parse_args()

final_fpr = 1.0
final_avg_fpr = 1.0
final_fpr1 = 1.0
final_fpr2 = 1.0
final_fpr3 = 1.0
final_fpr4 = 1.0
final_fpr5 = 1.0
final_fpr6 = 1.0
final_avg_auroc = -1.0
final_auroc1 = -1.0
final_auroc2 = -1.0
final_auroc3 = -1.0
final_auroc4 = -1.0
final_auroc5 = -1.0
final_auroc6 = -1.0
recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()
id_feats = []
id_feats_test = []
id_feats_eval = []
ood_feats = []
ood_feats_2 = []
ood_feats_3 = []
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

if args.dataset == "cifar10":
    if args.model == "resnet50":
        model = resnet.resnet50(num_classes=10)
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
    elif args.model == "wrn":
        model = WideResNet(40, 10, 2, dropRate=0.3)
        model.load_state_dict(torch.load("./ckpt/cifar10_wrn_pretrained_epoch_99.pt"))
elif args.dataset == "cifar100":
    if args.model == "resnet50":
        model = resnet.resnet50(num_classes=100)
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
    elif args.model == "wrn":
        model = WideResNet(40, 100, 2, dropRate=0.3)
        model.load_state_dict(torch.load("./ckpt/cifar100_wrn_pretrained_epoch_99.pt"))
model = model.cuda()

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

def ash_s_thre(x, threshold):
    # s1 = x.sum()
    k = (x >= threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    # s2 = x.sum()
    # scale = s1 / s2
    # x *= torch.exp(scale)
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

def extract_feats(feats, loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            feats.append(model.get_features_fc(data))

if args.dataset == "cifar10":
    if args.iter == 0:
        id_feats_test = torch.load("cifar10_densenet_id_feats_test_0")
        ood1_feats = torch.load("cifar10_densenet_ood1_feats_0")
        ood2_feats = torch.load("cifar10_densenet_ood2_feats_0")
        ood3_feats = torch.load("cifar10_densenet_ood3_feats_0")
        ood4_feats = torch.load("cifar10_densenet_ood4_feats_0")
        ood5_feats = torch.load("cifar10_densenet_ood5_feats_0")
        ood6_feats = torch.load("cifar10_densenet_ood6_feats_0")
    elif args.method == "pca":
        if args.iter == 10:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_pca_id_feats_test_10")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_pca_ood1_feats_10")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_pca_ood2_feats_10")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_pca_ood3_feats_10")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_pca_ood4_feats_10")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_pca_ood5_feats_10")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_pca_ood6_feats_10")
        elif args.iter == 20:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_pca_id_feats_test_20")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_pca_ood1_feats_20")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_pca_ood2_feats_20")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_pca_ood3_feats_20")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_pca_ood4_feats_20")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_pca_ood5_feats_20")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_pca_ood6_feats_20")
        elif args.iter == 50:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_pca_id_feats_test_50")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_pca_ood1_feats_50")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_pca_ood2_feats_50")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_pca_ood3_feats_50")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_pca_ood4_feats_50")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_pca_ood5_feats_50")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_pca_ood6_feats_50")
        elif args.iter == 100:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_pca_id_feats_test_100")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_pca_ood1_feats_100")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_pca_ood2_feats_100")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_pca_ood3_feats_100")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_pca_ood4_feats_100")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_pca_ood5_feats_100")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_pca_ood6_feats_100")
        elif args.iter == 200:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_pca_id_feats_test_200")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_pca_ood1_feats_200")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_pca_ood2_feats_200")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_pca_ood3_feats_200")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_pca_ood4_feats_200")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_pca_ood5_feats_200")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_pca_ood6_feats_200")
        elif args.iter == 500:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_pca_id_feats_test_500")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_pca_ood1_feats_500")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_pca_ood2_feats_500")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_pca_ood3_feats_500")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_pca_ood4_feats_500")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_pca_ood5_feats_500")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_pca_ood6_feats_500")
    elif args.method == "nmf":
        if args.iter == 10:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_nmf_id_feats_test_10")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood1_feats_10")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood2_feats_10")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood3_feats_10")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood4_feats_10")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood5_feats_10")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood6_feats_10")
        elif args.iter == 20:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_nmf_id_feats_test_20")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood1_feats_20")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood2_feats_20")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood3_feats_20")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood4_feats_20")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood5_feats_20")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood6_feats_20")
        elif args.iter == 50:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_nmf_id_feats_test_50")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood1_feats_50")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood2_feats_50")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood3_feats_50")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood4_feats_50")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood5_feats_50")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood6_feats_50")
        elif args.iter == 100:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_nmf_id_feats_test_100")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood1_feats_100")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood2_feats_100")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood3_feats_100")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood4_feats_100")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood5_feats_100")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood6_feats_100")
        elif args.iter == 200:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_nmf_id_feats_test_200")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood1_feats_200")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood2_feats_200")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood3_feats_200")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood4_feats_200")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood5_feats_200")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood6_feats_200")
        elif args.iter == 500:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_nmf_id_feats_test_500")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood1_feats_500")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood2_feats_500")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood3_feats_500")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood4_feats_500")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood5_feats_500")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_nmf_ood6_feats_500")
    elif args.method == "ica":
        if args.iter == 10:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_ica_id_feats_test_10")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_ica_ood1_feats_10")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_ica_ood2_feats_10")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_ica_ood3_feats_10")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_ica_ood4_feats_10")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_ica_ood5_feats_10")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_ica_ood6_feats_10")
        elif args.iter == 20:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_ica_id_feats_test_20")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_ica_ood1_feats_20")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_ica_ood2_feats_20")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_ica_ood3_feats_20")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_ica_ood4_feats_20")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_ica_ood5_feats_20")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_ica_ood6_feats_20")
        elif args.iter == 50:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_ica_id_feats_test_50")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_ica_ood1_feats_50")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_ica_ood2_feats_50")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_ica_ood3_feats_50")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_ica_ood4_feats_50")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_ica_ood5_feats_50")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_ica_ood6_feats_50")
        elif args.iter == 100:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_ica_id_feats_test_100")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_ica_ood1_feats_100")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_ica_ood2_feats_100")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_ica_ood3_feats_100")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_ica_ood4_feats_100")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_ica_ood5_feats_100")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_ica_ood6_feats_100")
        elif args.iter == 200:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_ica_id_feats_test_200")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_ica_ood1_feats_200")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_ica_ood2_feats_200")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_ica_ood3_feats_200")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_ica_ood4_feats_200")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_ica_ood5_feats_200")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_ica_ood6_feats_200")
        elif args.iter == 500:
            id_feats_test = torch.load("../data/cifar10_densenet_dice_ica_id_feats_test_500")
            ood1_feats = torch.load("../data/cifar10_densenet_dice_ica_ood1_feats_500")
            ood2_feats = torch.load("../data/cifar10_densenet_dice_ica_ood2_feats_500")
            ood3_feats = torch.load("../data/cifar10_densenet_dice_ica_ood3_feats_500")
            ood4_feats = torch.load("../data/cifar10_densenet_dice_ica_ood4_feats_500")
            ood5_feats = torch.load("../data/cifar10_densenet_dice_ica_ood5_feats_500")
            ood6_feats = torch.load("../data/cifar10_densenet_dice_ica_ood6_feats_500")
elif args.dataset == "cifar100":
    if args.iter == 0:
        id_feats_test = torch.load("cifar100_densenet_id_feats_test_0")
        ood1_feats = torch.load("cifar100_densenet_ood1_feats_0")
        ood2_feats = torch.load("cifar100_densenet_ood2_feats_0")
        ood3_feats = torch.load("cifar100_densenet_ood3_feats_0")
        ood4_feats = torch.load("cifar100_densenet_ood4_feats_0")
        ood5_feats = torch.load("cifar100_densenet_ood5_feats_0")
        ood6_feats = torch.load("cifar100_densenet_ood6_feats_0")
    elif args.method == "pca":
        if args.iter == 10:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_pca_id_feats_test_10")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_pca_ood1_feats_10")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_pca_ood2_feats_10")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_pca_ood3_feats_10")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_pca_ood4_feats_10")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_pca_ood5_feats_10")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_pca_ood6_feats_10")
        elif args.iter == 20:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_pca_id_feats_test_20")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_pca_ood1_feats_20")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_pca_ood2_feats_20")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_pca_ood3_feats_20")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_pca_ood4_feats_20")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_pca_ood5_feats_20")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_pca_ood6_feats_20")
        elif args.iter == 50:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_pca_id_feats_test_50")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_pca_ood1_feats_50")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_pca_ood2_feats_50")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_pca_ood3_feats_50")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_pca_ood4_feats_50")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_pca_ood5_feats_50")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_pca_ood6_feats_50")
        elif args.iter == 100:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_pca_id_feats_test_100")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_pca_ood1_feats_100")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_pca_ood2_feats_100")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_pca_ood3_feats_100")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_pca_ood4_feats_100")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_pca_ood5_feats_100")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_pca_ood6_feats_100")
        elif args.iter == 200:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_pca_id_feats_test_200")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_pca_ood1_feats_200")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_pca_ood2_feats_200")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_pca_ood3_feats_200")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_pca_ood4_feats_200")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_pca_ood5_feats_200")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_pca_ood6_feats_200")
        elif args.iter == 500:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_pca_id_feats_test_500")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_pca_ood1_feats_500")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_pca_ood2_feats_500")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_pca_ood3_feats_500")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_pca_ood4_feats_500")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_pca_ood5_feats_500")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_pca_ood6_feats_500")
    elif args.method == "nmf":
        if args.iter == 10:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_nmf_id_feats_test_10")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood1_feats_10")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood2_feats_10")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood3_feats_10")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood4_feats_10")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood5_feats_10")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood6_feats_10")
        elif args.iter == 20:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_nmf_id_feats_test_20")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood1_feats_20")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood2_feats_20")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood3_feats_20")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood4_feats_20")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood5_feats_20")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood6_feats_20")
        elif args.iter == 50:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_nmf_id_feats_test_50")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood1_feats_50")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood2_feats_50")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood3_feats_50")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood4_feats_50")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood5_feats_50")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood6_feats_50")
        elif args.iter == 100:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_nmf_id_feats_test_100")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood1_feats_100")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood2_feats_100")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood3_feats_100")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood4_feats_100")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood5_feats_100")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood6_feats_100")
        elif args.iter == 200:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_nmf_id_feats_test_200")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood1_feats_200")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood2_feats_200")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood3_feats_200")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood4_feats_200")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood5_feats_200")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood6_feats_200")
        elif args.iter == 500:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_nmf_id_feats_test_500")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood1_feats_500")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood2_feats_500")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood3_feats_500")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood4_feats_500")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood5_feats_500")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_nmf_ood6_feats_500")
    elif args.method == "ica":
        if args.iter == 10:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_ica_id_feats_test_10")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_ica_ood1_feats_10")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_ica_ood2_feats_10")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_ica_ood3_feats_10")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_ica_ood4_feats_10")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_ica_ood5_feats_10")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_ica_ood6_feats_10")
        elif args.iter == 20:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_ica_id_feats_test_20")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_ica_ood1_feats_20")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_ica_ood2_feats_20")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_ica_ood3_feats_20")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_ica_ood4_feats_20")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_ica_ood5_feats_20")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_ica_ood6_feats_20")
        elif args.iter == 50:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_ica_id_feats_test_50")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_ica_ood1_feats_50")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_ica_ood2_feats_50")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_ica_ood3_feats_50")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_ica_ood4_feats_50")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_ica_ood5_feats_50")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_ica_ood6_feats_50")
        elif args.iter == 100:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_ica_id_feats_test_100")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_ica_ood1_feats_100")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_ica_ood2_feats_100")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_ica_ood3_feats_100")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_ica_ood4_feats_100")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_ica_ood5_feats_100")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_ica_ood6_feats_100")
        elif args.iter == 200:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_ica_id_feats_test_200")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_ica_ood1_feats_200")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_ica_ood2_feats_200")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_ica_ood3_feats_200")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_ica_ood4_feats_200")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_ica_ood5_feats_200")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_ica_ood6_feats_200")
        elif args.iter == 500:
            id_feats_test = torch.load("../data/cifar100_densenet_dice_ica_id_feats_test_500")
            ood1_feats = torch.load("../data/cifar100_densenet_dice_ica_ood1_feats_500")
            ood2_feats = torch.load("../data/cifar100_densenet_dice_ica_ood2_feats_500")
            ood3_feats = torch.load("../data/cifar100_densenet_dice_ica_ood3_feats_500")
            ood4_feats = torch.load("../data/cifar100_densenet_dice_ica_ood4_feats_500")
            ood5_feats = torch.load("../data/cifar100_densenet_dice_ica_ood5_feats_500")
            ood6_feats = torch.load("../data/cifar100_densenet_dice_ica_ood6_feats_500")

# python insrect_patterns.py --model=densenet_dice --dataset=cifar10 --iter=500 --index=0 --method=nmf

# id_logits_test = model.fc(id_feats_test)
# ood1_logits = model.fc(ood1_feats)
# ood2_logits = model.fc(ood2_feats)
# ood3_logits = model.fc(ood3_feats)
# ood4_logits = model.fc(ood4_feats)
# ood5_logits = model.fc(ood5_feats)
# ood6_logits = model.fc(ood6_feats)

# id_scores_test = - torch.logsumexp(id_logits_test, axis=1).cpu().detach().numpy()
# ood1_scores = - torch.logsumexp(ood1_logits, axis=1).cpu().detach().numpy()
# ood2_scores = - torch.logsumexp(ood2_logits, axis=1).cpu().detach().numpy()
# ood3_scores = - torch.logsumexp(ood3_logits, axis=1).cpu().detach().numpy()
# ood4_scores = - torch.logsumexp(ood4_logits, axis=1).cpu().detach().numpy()
# ood5_scores = - torch.logsumexp(ood5_logits, axis=1).cpu().detach().numpy()
# ood6_scores = - torch.logsumexp(ood6_logits, axis=1).cpu().detach().numpy()

# _, index_id_test = torch.sort(torch.Tensor(id_scores_test))
# _, index_ood1 = torch.sort(torch.Tensor(ood1_scores))
# _, index_ood2 = torch.sort(torch.Tensor(ood2_scores))
# _, index_ood3 = torch.sort(torch.Tensor(ood3_scores))
# _, index_ood4 = torch.sort(torch.Tensor(ood4_scores))
# _, index_ood5 = torch.sort(torch.Tensor(ood5_scores))
# _, index_ood6 = torch.sort(torch.Tensor(ood6_scores))

# torch.save(index_id_test, "../data/nmf500_index_id_test.pkl")
# torch.save(index_ood1, "../data/nmf500_index_ood1.pkl")
# torch.save(index_ood2, "../data/nmf500_index_ood2.pkl")
# torch.save(index_ood3, "../data/nmf500_index_ood3.pkl")
# torch.save(index_ood4, "../data/nmf500_index_ood4.pkl")
# torch.save(index_ood5, "../data/nmf500_index_ood5.pkl")
# torch.save(index_ood6, "../data/nmf500_index_ood6.pkl")

index_id_test = torch.load("../data/nmf500_index_id_test.pkl")
index_ood1 = torch.load("../data/nmf500_index_ood1.pkl")
index_ood2 = torch.load("../data/nmf500_index_ood2.pkl")
index_ood3 = torch.load("../data/nmf500_index_ood3.pkl")
index_ood4 = torch.load("../data/nmf500_index_ood4.pkl")
index_ood5 = torch.load("../data/nmf500_index_ood5.pkl")
index_ood6 = torch.load("../data/nmf500_index_ood6.pkl")

colors = plt.get_cmap('Set1').colors

# plt.bar(range(342), id_feats_test[args.index].cpu(), color=colors[0])
plt.bar(range(342), id_feats_test[index_id_test[args.index]].cpu(), color=colors[0])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/id_test/" + args.dataset + "_" + args.model + "_id_feats_test_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/id_test/" + args.dataset + "_" + args.model + "_" + args.method + "_id_feats_test_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()

# plt.bar(range(342), ood1_feats[args.index].cpu(), color=colors[1])
plt.bar(range(342), ood1_feats[index_ood1[-(args.index + 1)]].cpu(), color=colors[1])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/ood1/" + args.dataset + "_" + args.model + "_ood1_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/ood1/" + args.dataset + "_" + args.model + "_" + args.method + "_ood1_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()

# plt.bar(range(342), ood2_feats[args.index].cpu(), color=colors[2])
plt.bar(range(342), ood2_feats[index_ood2[-(args.index + 1)]].cpu(), color=colors[2])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/ood2/" + args.dataset + "_" + args.model + "_ood2_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/ood2/" + args.dataset + "_" + args.model + "_" + args.method + "_ood2_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()

# plt.bar(range(342), ood3_feats[args.index].cpu(), color=colors[3])
plt.bar(range(342), ood3_feats[index_ood3[-(args.index + 1)]].cpu(), color=colors[3])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/ood3/" + args.dataset + "_" + args.model + "_ood3_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/ood3/" + args.dataset + "_" + args.model + "_" + args.method + "_ood3_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()

# plt.bar(range(342), ood4_feats[args.index].cpu(), color=colors[4])
plt.bar(range(342), ood4_feats[index_ood4[-(args.index + 1)]].cpu(), color=colors[4])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/ood4/" + args.dataset + "_" + args.model + "_ood4_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/ood4/" + args.dataset + "_" + args.model + "_" + args.method + "_ood4_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()

# plt.bar(range(342), ood5_feats[args.index].cpu(), color=colors[5])
plt.bar(range(342), ood5_feats[index_ood5[-(args.index + 1)]].cpu(), color=colors[5])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/ood5/" + args.dataset + "_" + args.model + "_ood5_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/ood5/" + args.dataset + "_" + args.model + "_" + args.method + "_ood5_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()

# plt.bar(range(342), ood6_feats[args.index].cpu(), color=colors[6])
plt.bar(range(342), ood6_feats[index_ood6[-(args.index + 1)]].cpu(), color=colors[6])
plt.ylim((-1, 3))
# plt.xticks([])
# plt.yticks([])
if args.method == None:
    plt.savefig("./patterns/ood6/" + args.dataset + "_" + args.model + "_ood6_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
else:
    plt.savefig("./patterns/ood6/" + args.dataset + "_" + args.model + "_" + args.method + "_ood6_feats_iter" + str(args.iter) + "_index" + str(args.index) + ".png")
plt.clf()
