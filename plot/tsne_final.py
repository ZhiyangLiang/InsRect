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
from sklearn.manifold import TSNE
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from utils.tinyimages_300K_random_loader import TinyImages
from models.wrn import WideResNet
import joblib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "wrn"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train_trans", type=str, choices=["id", "train"])
parser.add_argument("--auxiliary_trans", type=str, choices=["ood", "eval"])
parser.add_argument("--method", type=str, choices=["pca", "nmf", "ica"])
# parser.add_argument("--num_dataset", type=int)
parser.add_argument("--order", type=int)
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

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])

# if args.dataset == "cifar10":
#     if args.train_trans == "train":
#         id_data = dset.CIFAR10("../data/cifar10", train=True, transform=train_transform, download=False)
#     elif args.train_trans == "id":
#         id_data = dset.CIFAR10("../data/cifar10", train=True, transform=id_transform, download=False)
#     id_data_test = dset.CIFAR10("../data/cifar10", train=False, transform=id_transform, download=False)
    
#     if args.model == "resnet50":
#         model = resnet.resnet50(num_classes=10)
#         model.load_state_dict(torch.load("./ckpt/resnet50_cifar10-192-best-0.9546999931335449.pth"))
#     elif args.model == "densenet_dice":
#         model = DenseNet3(100, 10)
#         model.load_state_dict(torch.load("./ckpt/checkpoint_10.pth.tar")["state_dict"])
#     elif args.model == "wideresnet":
#         model = wideresnet(num_classes=10)
#         model.load_state_dict(torch.load("./ckpt/wideresnet_cifar10_epoch195_acc0.960599958896637.pt", map_location='cuda:0'))
#     elif args.model == "mobilenet":
#         model = mobilenet(class_num=10)
#         model.load_state_dict(torch.load("./ckpt/mobilenet_cifar10_epoch183_acc0.90829998254776.pt", map_location='cuda:0'))
#     elif args.model == "wrn":
#         model = WideResNet(40, 10, 2, dropRate=0.3)
#         model.load_state_dict(torch.load("./ckpt/cifar10_wrn_pretrained_epoch_99.pt"))

#     if args.method == "pca":
#         decomp = joblib.load("../data/pca_cifar10_comp2_densenet_dice_final.pkl")
#     elif args.method == "nmf":
#         decomp = joblib.load("../data/nmf_cifar10_comp2_densenet_dice_final.pkl")
#     elif args.method == "ica":
#         decomp = joblib.load("../data/ica_cifar10_comp2_densenet_dice_final.pkl")
#     elif args.method == "tsne":
#         decomp = joblib.load("../data/tsne_cifar10_comp2_densenet_dice_final.pkl")
# elif args.dataset == "cifar100":
#     if args.train_trans == "train":
#         id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
#     elif args.train_trans == "id":
#         id_data = dset.CIFAR100("../data/cifar100", train=True, transform=id_transform, download=False)
#     id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)

#     if args.model == "resnet50":
#         model = resnet.resnet50(num_classes=100)
#         model.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
#     elif args.model == "densenet_dice":
        # model = DenseNet3(100, 100)
        # model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
#     elif args.model == "wideresnet":
#         model = wideresnet(num_classes=100)
#         model.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
#     elif args.model == "mobilenet":
#         model = mobilenet(class_num=100)
#         model.load_state_dict(torch.load("./ckpt/mobilenet_epoch124_acc0.677299976348877.pt", map_location='cuda:0'))
#     elif args.model == "wrn":
#         model = WideResNet(40, 100, 2, dropRate=0.3)
#         model.load_state_dict(torch.load("./ckpt/cifar100_wrn_pretrained_epoch_99.pt"))

#     if args.method == "pca":
#         decomp = joblib.load("../data/pca_cifar100_comp2_densenet_dice_final.pkl")
#     elif args.method == "nmf":
#         decomp = joblib.load("../data/nmf_cifar100_comp2_densenet_dice_final.pkl")
#     elif args.method == "ica":
#         decomp = joblib.load("../data/ica_cifar100_comp2_densenet_dice_final.pkl")
#     elif args.method == "tsne":
#         decomp = joblib.load("../data/tsne_cifar100_comp2_densenet_dice_final.pkl")
# model = model.cuda()

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

# if args.auxiliary_trans == "ood":
#     ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
#     # ood_data = TinyImages(transform=ood_transform)
# elif args.auxiliary_trans == "eval":
#     ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=eval_transform)
#     # ood_data = TinyImages(transform=eval_transform)

# ood1_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
# ood2_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
# ood3_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
# ood4_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
# ood5_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
# ood6_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

# texture_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
# places365_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
# lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
# lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
# isun_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
# svhn_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

# id_loader = torch.utils.data.DataLoader(id_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
# id_loader_eval = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood1_loader = torch.utils.data.DataLoader(ood1_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood2_loader = torch.utils.data.DataLoader(ood2_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood3_loader = torch.utils.data.DataLoader(ood3_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood4_loader = torch.utils.data.DataLoader(ood4_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood5_loader = torch.utils.data.DataLoader(ood5_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# ood6_loader = torch.utils.data.DataLoader(ood6_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
# svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

model = DenseNet3(100, 100)
model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
model = model.cuda()

# acc = test(id_loader_test)
# print("acc: %.4f" % (acc))

# extract_feats(id_feats, id_loader)
# extract_feats(id_feats_test, id_loader_test)
# extract_feats(ood_feats, ood_loader)
# extract_feats(ood1_feats, ood1_loader)
# extract_feats(ood2_feats, ood2_loader)
# extract_feats(ood3_feats, ood3_loader)
# extract_feats(ood4_feats, ood4_loader)
# extract_feats(ood5_feats, ood5_loader)
# extract_feats(ood6_feats, ood6_loader)

# id_feats = torch.cat(id_feats, dim=0)
# id_feats_test = torch.cat(id_feats_test, dim=0)
# ood_feats = torch.cat(ood_feats, dim=0)[:100000]
# ood1_feats = torch.cat(ood1_feats, dim=0)
# ood2_feats = torch.cat(ood2_feats, dim=0)
# ood3_feats = torch.cat(ood3_feats, dim=0)
# ood4_feats = torch.cat(ood4_feats, dim=0)
# ood5_feats = torch.cat(ood5_feats, dim=0)
# ood6_feats = torch.cat(ood6_feats, dim=0)

# id_feats = torch.load("../data/final_cifar100_densenet_dice_id_feats.pkl")
# id_feats_test = torch.load("../data/final_cifar100_densenet_dice_id_feats_test.pkl")
# ood_feats = torch.load("../data/final_cifar100_densenet_dice_ood_feats.pkl")
# ood1_feats = torch.load("../data/final_cifar100_densenet_dice_ood1_feats.pkl")
# ood2_feats = torch.load("../data/final_cifar100_densenet_dice_ood2_feats.pkl")
# ood3_feats = torch.load("../data/final_cifar100_densenet_dice_ood3_feats.pkl")
# ood4_feats = torch.load("../data/final_cifar100_densenet_dice_ood4_feats.pkl")
# ood5_feats = torch.load("../data/final_cifar100_densenet_dice_ood5_feats.pkl")
# ood6_feats = torch.load("../data/final_cifar100_densenet_dice_ood6_feats.pkl")

# # id_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_id_feats.pkl")
# id_feats_test = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_id_feats_test.pkl")
# # ood_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood_feats.pkl")
# ood1_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood1_feats.pkl")
# ood2_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood2_feats.pkl")
# ood3_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood3_feats.pkl")
# ood4_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood4_feats.pkl")
# ood5_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood5_feats.pkl")
# ood6_feats = torch.load("../data/after_clamping_sca_cifar100_densenet_dice_ood6_feats.pkl")

if args.method == "pca":
    id_feats_test = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_id_feats_test.pkl")
    ood1_feats = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_ood1_feats.pkl")
    ood2_feats = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_ood2_feats.pkl")
    ood3_feats = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_ood3_feats.pkl")
    ood4_feats = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_ood4_feats.pkl")
    ood5_feats = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_ood5_feats.pkl")
    ood6_feats = torch.load("../data/pca_after_clamping_sca_cifar100_densenet_dice_ood6_feats.pkl")
elif args.method == "nmf":
    id_feats_test = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_id_feats_test.pkl")
    ood1_feats = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_ood1_feats.pkl")
    ood2_feats = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_ood2_feats.pkl")
    ood3_feats = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_ood3_feats.pkl")
    ood4_feats = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_ood4_feats.pkl")
    ood5_feats = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_ood5_feats.pkl")
    ood6_feats = torch.load("../data/nmf_after_clamping_sca_cifar100_densenet_dice_ood6_feats.pkl")
elif args.method == "ica":
    id_feats_test = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_id_feats_test.pkl")
    ood1_feats = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_ood1_feats.pkl")
    ood2_feats = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_ood2_feats.pkl")
    ood3_feats = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_ood3_feats.pkl")
    ood4_feats = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_ood4_feats.pkl")
    ood5_feats = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_ood5_feats.pkl")
    ood6_feats = torch.load("../data/ica_after_clamping_sca_cifar100_densenet_dice_ood6_feats.pkl")

id_index = torch.load("../data/id_index_test")
ood1_index = torch.load("../data/ood1_index")
ood2_index = torch.load("../data/ood2_index")
ood3_index = torch.load("../data/ood3_index")
ood4_index = torch.load("../data/ood4_index")
ood5_index = torch.load("../data/ood5_index")
ood6_index = torch.load("../data/ood6_index")

if args.order == 1:
    id_feats_test = id_feats_test[id_index[1000:1500]]
    ood1_feats = ood1_feats[ood1_index[-1500:-1000]]
    ood2_feats = ood2_feats[ood2_index[-1500:-1000]]
    ood3_feats = ood3_feats[ood3_index[-1500:-1000]]
    ood4_feats = ood4_feats[ood4_index[-1500:-1000]]
    ood5_feats = ood5_feats[ood5_index[-1500:-1000]]
    ood6_feats = ood6_feats[ood6_index[-1500:-1000]]
elif args.order == 2:
    id_feats_test = id_feats_test[id_index[1500:2000]]
    ood1_feats = ood1_feats[ood1_index[-2000:-1500]]
    ood2_feats = ood2_feats[ood2_index[-2000:-1500]]
    ood3_feats = ood3_feats[ood3_index[-2000:-1500]]
    ood4_feats = ood4_feats[ood4_index[-2000:-1500]]
    ood5_feats = ood5_feats[ood5_index[-2000:-1500]]
    ood6_feats = ood6_feats[ood6_index[-2000:-1500]]
elif args.order == 3:
    id_feats_test = id_feats_test[id_index[500:1000]]
    ood1_feats = ood1_feats[ood1_index[-1000:-500]]
    ood2_feats = ood2_feats[ood2_index[-1000:-500]]
    ood3_feats = ood3_feats[ood3_index[-1000:-500]]
    ood4_feats = ood4_feats[ood4_index[-1000:-500]]
    ood5_feats = ood5_feats[ood5_index[-1000:-500]]
    ood6_feats = ood6_feats[ood6_index[-1000:-500]]
elif args.order == 4:
    id_feats_test = id_feats_test[id_index[:500]]
    ood1_feats = ood1_feats[ood1_index[-500:]]
    ood2_feats = ood2_feats[ood2_index[-500:]]
    ood3_feats = ood3_feats[ood3_index[-500:]]
    ood4_feats = ood4_feats[ood4_index[-500:]]
    ood5_feats = ood5_feats[ood5_index[-500:]]
    ood6_feats = ood6_feats[ood6_index[-500:]]
elif args.order == 5:
    id_feats_test = id_feats_test[id_index[2000:2500]]
    ood1_feats = ood1_feats[ood1_index[-2500:-2000]]
    ood2_feats = ood2_feats[ood2_index[-2500:-2000]]
    ood3_feats = ood3_feats[ood3_index[-2500:-2000]]
    ood4_feats = ood4_feats[ood4_index[-2500:-2000]]
    ood5_feats = ood5_feats[ood5_index[-2500:-2000]]
    ood6_feats = ood6_feats[ood6_index[-2500:-2000]]

# tsne_data = torch.cat([id_feats_test, ood1_feats, ood2_feats, ood3_feats, ood4_feats, ood5_feats, ood6_feats])
# decomp = TSNE(n_components=2, n_iter=20000)
# tsne_data_trans = decomp.fit_transform(tsne_data.cpu())

if args.method == "pca":
    # if args.order == 3:
    #     torch.save(tsne_data_trans, "../data/cifar100_pca_tsne_3.pkl")
    # elif args.order == 4:
    #     torch.save(tsne_data_trans, "../data/cifar100_pca_tsne_4.pkl")
    # elif args.order == 5:
    #     torch.save(tsne_data_trans, "../data/cifar100_pca_tsne_5.pkl")
    tsne_data_trans = torch.load("../data/cifar100_pca_tsne.pkl")
elif args.method == "nmf":
    # if args.order == 3:
    #     torch.save(tsne_data_trans, "../data/cifar100_nmf_tsne_3.pkl")
    # elif args.order == 4:
    #     torch.save(tsne_data_trans, "../data/cifar100_nmf_tsne_4.pkl")
    # elif args.order == 5:
    #     torch.save(tsne_data_trans, "../data/cifar100_nmf_tsne_5.pkl")
    tsne_data_trans = torch.load("../data/cifar100_nmf_tsne_4.pkl")
elif args.method == "ica":
    # if args.order == 3:
    #     torch.save(tsne_data_trans, "../data/cifar100_ica_tsne_3.pkl")
    # elif args.order == 4:
    #     torch.save(tsne_data_trans, "../data/cifar100_ica_tsne_4.pkl")
    # elif args.order == 5:
    #     torch.save(tsne_data_trans, "../data/cifar100_ica_tsne_5.pkl")
    tsne_data_trans = torch.load("../data/cifar100_ica_tsne.pkl")

# plt.figure(figsize=(12, 8))
# colors = plt.get_cmap('Set1').colors
# for i in range(7):
#     plt.scatter(tsne_data_trans[i *500:(i + 1) * 500, 0], tsne_data_trans[i * 500:(i + 1) * 500, 1], s=5, color=colors[i], marker='o')

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))
plt.scatter(tsne_data_trans[:500, 0], tsne_data_trans[:500, 1], s=5, color=['royalblue'], marker='o')
plt.scatter(tsne_data_trans[1000:1500, 0], tsne_data_trans[1000:1500, 1], s=5, color=['dimgrey'], marker='o')

plt.xlim((-100, 100))
plt.ylim((-100, 100))
plt.xticks([-100, -50, 0, 50, 100])
plt.yticks([-100, -50, 0, 50, 100])
plt.tick_params(labelsize=25)
plt.rcParams.update({'font.size': 25})
# plt.legend(loc="upper right")
# plt.legend()
# plt.show()
# plt.savefig("./tsne/scatter_ood" + str(args.num_dataset) + ".png")
# plt.savefig("./tsne/clamp_scatter.png")
# plt.savefig("./tsne/final_scatter_2000.png")
# plt.savefig("./tsne/final_scatter_2000.svg")
# plt.savefig("./tsne/final_all_scatter_2000.png")
# plt.savefig("./tsne/final_all_scatter_2000.svg")

if args.method == "pca":
    # if args.order == 1:
    #     plt.savefig("./decomp/final_pca_scatter_500_1_all.png")
    # elif args.order == 2:
    #     plt.savefig("./decomp/final_pca_scatter_500_2_all.svg")
    # elif args.order == 3:
    #     plt.savefig("./decomp/final_pca_scatter_500_3_all.png")
    # elif args.order == 4:
    #     plt.savefig("./decomp/final_pca_scatter_500_4_all.png")
    # elif args.order == 5:
    #     plt.savefig("./decomp/final_pca_scatter_500_5_all.png")
    plt.savefig("./decomp/new_final_pca_scatter.svg")
elif args.method == "nmf":
    # if args.order == 1:
    #     plt.savefig("./decomp/final_nmf_scatter_500_1_all.png")
    # elif args.order == 2:
    #     plt.savefig("./decomp/final_nmf_scatter_500_2_all.svg")
    # elif args.order == 3:
    #     plt.savefig("./decomp/final_nmf_scatter_500_3_all.png")
    # elif args.order == 4:
    #     plt.savefig("./decomp/final_nmf_scatter_500_4_all.png")
    # elif args.order == 5:
    #     plt.savefig("./decomp/final_nmf_scatter_500_5_all.png")
    plt.savefig("./decomp/new_final_nmf_scatter.svg")
elif args.method == "ica":
    # if args.order == 1:
    #     plt.savefig("./decomp/final_ica_scatter_500_1_all.png")
    # elif args.order == 2:
    #     plt.savefig("./decomp/final_ica_scatter_500_2_all.png")
    # elif args.order == 3:
    #     plt.savefig("./decomp/final_ica_scatter_500_3_all.png")
    # elif args.order == 4:
    #     plt.savefig("./decomp/final_ica_scatter_500_4_all.png")
    # elif args.order == 5:
    #     plt.savefig("./decomp/final_ica_scatter_500_5_all.png")
    plt.savefig("./decomp/new_final_ica_scatter.svg")
