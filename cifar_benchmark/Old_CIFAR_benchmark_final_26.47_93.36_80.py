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

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "wrn"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train_trans", type=str, choices=["id", "train"])
parser.add_argument("--auxiliary_trans", type=str, choices=["ood", "eval"])
parser.add_argument("--num_component", type=int)
parser.add_argument("--acquisition", type=str, choices=["ucb", "ei", "poi"])
parser.add_argument("--kappa", type=float)
parser.add_argument("--xi", type=float)
parser.add_argument("--lower_percentile", type=float)
parser.add_argument("--upper_percentile", type=float)
parser.add_argument("--ash_percentile", type=float)
parser.add_argument("--metric", type=str, choices=["fpr", "auroc", "both"])
parser.add_argument("--percent", type=int)
parser.add_argument("--seed", type=int)
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
    if args.train_trans == "train":
        id_data = dset.CIFAR10("../data/cifar10", train=True, transform=train_transform, download=False)
    elif args.train_trans == "id":
        id_data = dset.CIFAR10("../data/cifar10", train=True, transform=id_transform, download=False)
    id_data_test = dset.CIFAR10("../data/cifar10", train=False, transform=id_transform, download=False)
    
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
    if args.train_trans == "train":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
    elif args.train_trans == "id":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=id_transform, download=False)
    id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)

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

if args.auxiliary_trans == "ood":
    ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
    # ood_data = TinyImages(transform=ood_transform)
elif args.auxiliary_trans == "eval":
    ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=eval_transform)
    # ood_data = TinyImages(transform=eval_transform)

ood1_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
ood2_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
ood3_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
ood4_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
ood5_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
ood6_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

texture_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
places365_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
isun_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
svhn_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

id_loader = torch.utils.data.DataLoader(id_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
id_loader_eval = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood1_loader = torch.utils.data.DataLoader(ood1_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood2_loader = torch.utils.data.DataLoader(ood2_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood3_loader = torch.utils.data.DataLoader(ood3_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood4_loader = torch.utils.data.DataLoader(ood4_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood5_loader = torch.utils.data.DataLoader(ood5_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood6_loader = torch.utils.data.DataLoader(ood6_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

acc = test(id_loader_test)
print("acc: %.4f" % (acc))

extract_feats(id_feats, id_loader)
extract_feats(id_feats_test, id_loader_test)
extract_feats(ood_feats, ood_loader)
extract_feats(ood1_feats, ood1_loader)
extract_feats(ood2_feats, ood2_loader)
extract_feats(ood3_feats, ood3_loader)
extract_feats(ood4_feats, ood4_loader)
extract_feats(ood5_feats, ood5_loader)
extract_feats(ood6_feats, ood6_loader)

id_feats = torch.cat(id_feats, dim=0)
id_feats_test = torch.cat(id_feats_test, dim=0)
ood_feats = torch.cat(ood_feats, dim=0)[:100000]
ood1_feats = torch.cat(ood1_feats, dim=0)
ood2_feats = torch.cat(ood2_feats, dim=0)
ood3_feats = torch.cat(ood3_feats, dim=0)
ood4_feats = torch.cat(ood4_feats, dim=0)
ood5_feats = torch.cat(ood5_feats, dim=0)
ood6_feats = torch.cat(ood6_feats, dim=0)

nmf = NMF(n_components=args.num_component, max_iter=20000)
# nmf = PCA(n_components=args.num_component)
# nmf = FastICA(n_components=args.num_component, max_iter=20000)
nmf.fit(id_feats.cpu())
ash_bound = np.percentile(nmf.components_, args.ash_percentile)
react_lower_bound = np.percentile(nmf.components_, args.lower_percentile)
react_upper_bound = np.percentile(nmf.components_, args.upper_percentile)

nmf_components = torch.Tensor(nmf.components_)
nmf_id_feats = torch.Tensor(nmf.transform(id_feats.cpu()))
nmf_ood_feats = torch.Tensor(nmf.transform(ood_feats.cpu()))
nmf_id_feats_test = torch.Tensor(nmf.transform(id_feats_test.cpu()))
nmf_ood1_feats = torch.Tensor(nmf.transform(ood1_feats.cpu()))
nmf_ood2_feats = torch.Tensor(nmf.transform(ood2_feats.cpu()))
nmf_ood3_feats = torch.Tensor(nmf.transform(ood3_feats.cpu()))
nmf_ood4_feats = torch.Tensor(nmf.transform(ood4_feats.cpu()))
nmf_ood5_feats = torch.Tensor(nmf.transform(ood5_feats.cpu()))
nmf_ood6_feats = torch.Tensor(nmf.transform(ood6_feats.cpu()))

# 5
# def eval_datasets(r0, r1, r2, r3, r4, c0, c1, c2, c3, c4):
# 10
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
# 15
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14):
# 20
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19):
# 25
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24):
# 30
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29):
# 35
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34):
# 40
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39):
# 45
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44):
# 50
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49):
# 55
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54):
# 60
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59):
# 65
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64):
# 70
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69):
# 75
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74):
# 80
def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79):
# 100
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99):
# 150
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146, r147, r148, r149, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149):
# 200
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146, r147, r148, r149, r150, r151, r152, r153, r154, r155, r156, r157, r158, r159, r160, r161, r162, r163, r164, r165, r166, r167, r168, r169, r170, r171, r172, r173, r174, r175, r176, r177, r178, r179, r180, r181, r182, r183, r184, r185, r186, r187, r188, r189, r190, r191, r192, r193, r194, r195, r196, r197, r198, r199, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149, c150, c151, c152, c153, c154, c155, c156, c157, c158, c159, c160, c161, c162, c163, c164, c165, c166, c167, c168, c169, c170, c171, c172, c173, c174, c175, c176, c177, c178, c179, c180, c181, c182, c183, c184, c185, c186, c187, c188, c189, c190, c191, c192, c193, c194, c195, c196, c197, c198, c199):
    global id_feats, id_feats_test, ood_feats, ood1_feats, ood2_feats, ood3_feats, ood4_feats, ood5_feats, ood6_feats
    global final_fpr, final_fpr1, final_fpr2, final_fpr3, final_fpr4, final_fpr5, final_fpr6, final_avg_fpr
    global final_auroc, final_auroc1, final_auroc2, final_auroc3, final_auroc4, final_auroc5, final_auroc6, final_avg_auroc
    m_id_feats = torch.clone(id_feats)
    m_id_feats_test = torch.clone(id_feats_test)
    m_ood_feats = torch.clone(ood_feats)
    m_ood1_feats = torch.clone(ood1_feats)
    m_ood2_feats = torch.clone(ood2_feats)
    m_ood3_feats = torch.clone(ood3_feats)
    m_ood4_feats = torch.clone(ood4_feats)
    m_ood5_feats = torch.clone(ood5_feats)
    m_ood6_feats = torch.clone(ood6_feats)

    m_id_error = m_id_feats - torch.Tensor(nmf.transform(m_id_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    m_ood_error = m_ood_feats - torch.Tensor(nmf.transform(m_ood_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    m_trans = torch.clone(torch.Tensor(nmf.components_))
    if args.num_component == 5:
        react_list = [r0, r1, r2, r3, r4]
        ash_list = [c0, c1, c2, c3, c4]
    elif args.num_component == 10:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    elif args.num_component == 15:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]
    elif args.num_component == 20:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19]
    elif args.num_component == 25:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24]
    elif args.num_component == 30:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29]
    elif args.num_component == 35:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34]
    elif args.num_component == 40:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39]
    elif args.num_component == 45:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44]
    elif args.num_component == 50:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49]
    elif args.num_component == 55:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54]
    elif args.num_component == 60:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59]
    elif args.num_component == 65:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64]
    elif args.num_component == 70:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69]
    elif args.num_component == 75:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74]
    elif args.num_component == 80:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79]
    elif args.num_component == 100:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99]
    elif args.num_component == 150:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146, r147, r148, r149]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149]
    elif args.num_component == 200:
        react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146, r147, r148, r149, r150, r151, r152, r153, r154, r155, r156, r157, r158, r159, r160, r161, r162, r163, r164, r165, r166, r167, r168, r169, r170, r171, r172, r173, r174, r175, r176, r177, r178, r179, r180, r181, r182, r183, r184, r185, r186, r187, r188, r189, r190, r191, r192, r193, r194, r195, r196, r197, r198, r199]
        ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149, c150, c151, c152, c153, c154, c155, c156, c157, c158, c159, c160, c161, c162, c163, c164, c165, c166, c167, c168, c169, c170, c171, c172, c173, c174, c175, c176, c177, c178, c179, c180, c181, c182, c183, c184, c185, c186, c187, c188, c189, c190, c191, c192, c193, c194, c195, c196, c197, c198, c199]

    for (i, j) in zip(range(m_trans.shape[0]), react_list):
        m_trans[i] = react(m_trans[i], j)

    for (i, j) in zip(range(m_trans.shape[0]), ash_list):
        m_trans[i] = ash_s_thre(m_trans[i], j)

    m_id_feats = torch.Tensor(nmf.transform(m_id_feats.cpu())).mm(m_trans).cuda() + m_id_error
    m_ood_feats = torch.Tensor(nmf.transform(m_ood_feats.cpu())).mm(m_trans).cuda() + m_ood_error

    if args.model == "resnet50" or args.model == "densenet_dice" or args.model == "mobilenet" or args.model == "wrn":
        m_id_logits = model.fc(m_id_feats)
        m_ood_logits = model.fc(m_ood_feats)
    elif args.model == "wideresnet":
        m_id_logits = model.linear(m_id_feats)
        m_ood_logits = model.linear(m_ood_feats)

    m_id_score =  - torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
    m_ood_score =  - torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()
    if torch.isnan(torch.Tensor(m_id_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood_score)).sum() != 0:
        return -1e9
        # raise ValueError("The tensor has nan or inf elements")
    
    fpr, auroc, aupr = score_get_and_print_results(log, m_id_score, m_ood_score)
    if fpr < final_fpr:
        m_id_error_test = m_id_feats_test - torch.Tensor(nmf.transform(m_id_feats_test.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood1_error = m_ood1_feats - torch.Tensor(nmf.transform(m_ood1_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood2_error = m_ood2_feats - torch.Tensor(nmf.transform(m_ood2_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood3_error = m_ood3_feats - torch.Tensor(nmf.transform(m_ood3_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood4_error = m_ood4_feats - torch.Tensor(nmf.transform(m_ood4_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood5_error = m_ood5_feats - torch.Tensor(nmf.transform(m_ood5_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood6_error = m_ood6_feats - torch.Tensor(nmf.transform(m_ood6_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()

        m_id_feats_test = torch.Tensor(nmf.transform(m_id_feats_test.cpu())).mm(m_trans).cuda() + m_id_error_test
        m_ood1_feats = torch.Tensor(nmf.transform(m_ood1_feats.cpu())).mm(m_trans).cuda() + m_ood1_error
        m_ood2_feats = torch.Tensor(nmf.transform(m_ood2_feats.cpu())).mm(m_trans).cuda() + m_ood2_error
        m_ood3_feats = torch.Tensor(nmf.transform(m_ood3_feats.cpu())).mm(m_trans).cuda() + m_ood3_error
        m_ood4_feats = torch.Tensor(nmf.transform(m_ood4_feats.cpu())).mm(m_trans).cuda() + m_ood4_error
        m_ood5_feats = torch.Tensor(nmf.transform(m_ood5_feats.cpu())).mm(m_trans).cuda() + m_ood5_error
        m_ood6_feats = torch.Tensor(nmf.transform(m_ood6_feats.cpu())).mm(m_trans).cuda() + m_ood6_error

        m_id_feats_test = ash_s(m_id_feats_test, args.percent)
        m_ood1_feats = ash_s(m_ood1_feats, args.percent)
        m_ood2_feats = ash_s(m_ood2_feats, args.percent)
        m_ood3_feats = ash_s(m_ood3_feats, args.percent)
        m_ood4_feats = ash_s(m_ood4_feats, args.percent)
        m_ood5_feats = ash_s(m_ood5_feats, args.percent)
        m_ood6_feats = ash_s(m_ood6_feats, args.percent)

        if args.model == "resnet50" or args.model == "densenet_dice" or args.model == "mobilenet" or args.model == "wrn":
            m_id_logits_test = model.fc(m_id_feats_test)
            m_ood1_logits = model.fc(m_ood1_feats)
            m_ood2_logits = model.fc(m_ood2_feats)
            m_ood3_logits = model.fc(m_ood3_feats)
            m_ood4_logits = model.fc(m_ood4_feats)
            m_ood5_logits = model.fc(m_ood5_feats)
            m_ood6_logits = model.fc(m_ood6_feats)
        elif args.model == "wideresnet":
            m_id_logits_test = model.linear(m_id_feats_test)
            m_ood1_logits = model.linear(m_ood1_feats)
            m_ood2_logits = model.linear(m_ood2_feats)
            m_ood3_logits = model.linear(m_ood3_feats)
            m_ood4_logits = model.linear(m_ood4_feats)
            m_ood5_logits = model.linear(m_ood5_feats)
            m_ood6_logits = model.linear(m_ood6_feats)

        m_id_score_test =  - torch.logsumexp(m_id_logits_test, axis=1).cpu().detach().numpy()
        m_ood1_score =  - torch.logsumexp(m_ood1_logits, axis=1).cpu().detach().numpy()
        m_ood2_score =  - torch.logsumexp(m_ood2_logits, axis=1).cpu().detach().numpy()
        m_ood3_score =  - torch.logsumexp(m_ood3_logits, axis=1).cpu().detach().numpy()
        m_ood4_score =  - torch.logsumexp(m_ood4_logits, axis=1).cpu().detach().numpy()
        m_ood5_score =  - torch.logsumexp(m_ood5_logits, axis=1).cpu().detach().numpy()
        m_ood6_score =  - torch.logsumexp(m_ood6_logits, axis=1).cpu().detach().numpy()
        if torch.isnan(torch.Tensor(m_id_score_test)).sum() != 0 or torch.isnan(torch.Tensor(m_ood1_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood2_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood3_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood4_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood5_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood6_score)).sum() != 0:
            return -1e9
            # raise ValueError("The tensor has nan or inf elements")

        fpr1, auroc1, aupr1 = score_get_and_print_results(log, m_id_score_test, m_ood1_score)
        fpr2, auroc2, aupr2 = score_get_and_print_results(log, m_id_score_test, m_ood2_score)
        fpr3, auroc3, aupr3 = score_get_and_print_results(log, m_id_score_test, m_ood3_score)
        fpr4, auroc4, aupr4 = score_get_and_print_results(log, m_id_score_test, m_ood4_score)
        fpr5, auroc5, aupr5 = score_get_and_print_results(log, m_id_score_test, m_ood5_score)
        fpr6, auroc6, aupr6 = score_get_and_print_results(log, m_id_score_test, m_ood6_score)
        avg_fpr = (fpr1 + fpr2 + fpr3 + fpr4 + fpr5 + fpr6) / 6
        avg_auroc = (auroc1 + auroc2 + auroc3 + auroc4 + auroc5 + auroc6) / 6

        final_fpr = fpr
        final_avg_fpr = avg_fpr
        final_fpr1 = fpr1
        final_fpr2 = fpr2
        final_fpr3 = fpr3
        final_fpr4 = fpr4
        final_fpr5 = fpr5
        final_fpr6 = fpr6

        final_auroc = auroc
        final_avg_auroc = avg_auroc
        final_auroc1 = auroc1
        final_auroc2 = auroc2
        final_auroc3 = auroc3
        final_auroc4 = auroc4
        final_auroc5 = auroc5
        final_auroc6 = auroc6
    print("final_fpr: %.4f; final_auroc: %.4f" % (final_fpr, final_auroc))
    print("texture_fpr: %.4f; texture_auroc: %.4f" % (final_fpr1, final_auroc1))
    print("places365_fpr: %.4f; places365_auroc: %.4f" % (final_fpr2, final_auroc2))
    print("lsunr_fpr: %.4f; lsunr_auroc: %.4f" % (final_fpr3, final_auroc3))
    print("lsunc_fpr: %.4f; lsunc_auroc: %.4f" % (final_fpr4, final_auroc4))
    print("isun_fpr: %.4f; isun_auroc: %.4f" % (final_fpr5, final_auroc5))
    print("svhn_fpr: %.4f; svhn_auroc: %.4f" % (final_fpr6, final_auroc6))
    print("final_avg_fpr: %.4f; final_avg_auroc: %.4f" % (final_avg_fpr, final_avg_auroc))
    if args.metric == "fpr":
        return - fpr
    elif args.metric == "auroc":
        return auroc
    elif args.metric == "both":
        return auroc - fpr

ood_bayesian = BayesianOptimization(
    eval_datasets,
    {
    'r0': (react_lower_bound, react_upper_bound),
    'r1': (react_lower_bound, react_upper_bound),
    'r2': (react_lower_bound, react_upper_bound),
    'r3': (react_lower_bound, react_upper_bound),
    'r4': (react_lower_bound, react_upper_bound),
    'r5': (react_lower_bound, react_upper_bound),
    'r6': (react_lower_bound, react_upper_bound),
    'r7': (react_lower_bound, react_upper_bound),
    'r8': (react_lower_bound, react_upper_bound),
    'r9': (react_lower_bound, react_upper_bound),
    'r10': (react_lower_bound, react_upper_bound),
    'r11': (react_lower_bound, react_upper_bound),
    'r12': (react_lower_bound, react_upper_bound),
    'r13': (react_lower_bound, react_upper_bound),
    'r14': (react_lower_bound, react_upper_bound),
    'r15': (react_lower_bound, react_upper_bound),
    'r16': (react_lower_bound, react_upper_bound),
    'r17': (react_lower_bound, react_upper_bound),
    'r18': (react_lower_bound, react_upper_bound),
    'r19': (react_lower_bound, react_upper_bound),
    'r20': (react_lower_bound, react_upper_bound),
    'r21': (react_lower_bound, react_upper_bound),
    'r22': (react_lower_bound, react_upper_bound),
    'r23': (react_lower_bound, react_upper_bound),
    'r24': (react_lower_bound, react_upper_bound),
    'r25': (react_lower_bound, react_upper_bound),
    'r26': (react_lower_bound, react_upper_bound),
    'r27': (react_lower_bound, react_upper_bound),
    'r28': (react_lower_bound, react_upper_bound),
    'r29': (react_lower_bound, react_upper_bound),
    'r30': (react_lower_bound, react_upper_bound),
    'r31': (react_lower_bound, react_upper_bound),
    'r32': (react_lower_bound, react_upper_bound),
    'r33': (react_lower_bound, react_upper_bound),
    'r34': (react_lower_bound, react_upper_bound),
    'r35': (react_lower_bound, react_upper_bound),
    'r36': (react_lower_bound, react_upper_bound),
    'r37': (react_lower_bound, react_upper_bound),
    'r38': (react_lower_bound, react_upper_bound),
    'r39': (react_lower_bound, react_upper_bound),
    'r40': (react_lower_bound, react_upper_bound),
    'r41': (react_lower_bound, react_upper_bound),
    'r42': (react_lower_bound, react_upper_bound),
    'r43': (react_lower_bound, react_upper_bound),
    'r44': (react_lower_bound, react_upper_bound),
    'r45': (react_lower_bound, react_upper_bound),
    'r46': (react_lower_bound, react_upper_bound),
    'r47': (react_lower_bound, react_upper_bound),
    'r48': (react_lower_bound, react_upper_bound),
    'r49': (react_lower_bound, react_upper_bound),
    'r50': (react_lower_bound, react_upper_bound),
    'r51': (react_lower_bound, react_upper_bound),
    'r52': (react_lower_bound, react_upper_bound),
    'r53': (react_lower_bound, react_upper_bound),
    'r54': (react_lower_bound, react_upper_bound),
    'r55': (react_lower_bound, react_upper_bound),
    'r56': (react_lower_bound, react_upper_bound),
    'r57': (react_lower_bound, react_upper_bound),
    'r58': (react_lower_bound, react_upper_bound),
    'r59': (react_lower_bound, react_upper_bound),
    'r60': (react_lower_bound, react_upper_bound),
    'r61': (react_lower_bound, react_upper_bound),
    'r62': (react_lower_bound, react_upper_bound),
    'r63': (react_lower_bound, react_upper_bound),
    'r64': (react_lower_bound, react_upper_bound),
    'r65': (react_lower_bound, react_upper_bound),
    'r66': (react_lower_bound, react_upper_bound),
    'r67': (react_lower_bound, react_upper_bound),
    'r68': (react_lower_bound, react_upper_bound),
    'r69': (react_lower_bound, react_upper_bound),
    'r70': (react_lower_bound, react_upper_bound),
    'r71': (react_lower_bound, react_upper_bound),
    'r72': (react_lower_bound, react_upper_bound),
    'r73': (react_lower_bound, react_upper_bound),
    'r74': (react_lower_bound, react_upper_bound),
    'r75': (react_lower_bound, react_upper_bound),
    'r76': (react_lower_bound, react_upper_bound),
    'r77': (react_lower_bound, react_upper_bound),
    'r78': (react_lower_bound, react_upper_bound),
    'r79': (react_lower_bound, react_upper_bound),
    # 'r80': (react_lower_bound, react_upper_bound),
    # 'r81': (react_lower_bound, react_upper_bound),
    # 'r82': (react_lower_bound, react_upper_bound),
    # 'r83': (react_lower_bound, react_upper_bound),
    # 'r84': (react_lower_bound, react_upper_bound),
    # 'r85': (react_lower_bound, react_upper_bound),
    # 'r86': (react_lower_bound, react_upper_bound),
    # 'r87': (react_lower_bound, react_upper_bound),
    # 'r88': (react_lower_bound, react_upper_bound),
    # 'r89': (react_lower_bound, react_upper_bound),
    # 'r90': (react_lower_bound, react_upper_bound),
    # 'r91': (react_lower_bound, react_upper_bound),
    # 'r92': (react_lower_bound, react_upper_bound),
    # 'r93': (react_lower_bound, react_upper_bound),
    # 'r94': (react_lower_bound, react_upper_bound),
    # 'r95': (react_lower_bound, react_upper_bound),
    # 'r96': (react_lower_bound, react_upper_bound),
    # 'r97': (react_lower_bound, react_upper_bound),
    # 'r98': (react_lower_bound, react_upper_bound),
    # 'r99': (react_lower_bound, react_upper_bound),
    # 'r100': (react_lower_bound, react_upper_bound),
    # 'r101': (react_lower_bound, react_upper_bound),
    # 'r102': (react_lower_bound, react_upper_bound),
    # 'r103': (react_lower_bound, react_upper_bound),
    # 'r104': (react_lower_bound, react_upper_bound),
    # 'r105': (react_lower_bound, react_upper_bound),
    # 'r106': (react_lower_bound, react_upper_bound),
    # 'r107': (react_lower_bound, react_upper_bound),
    # 'r108': (react_lower_bound, react_upper_bound),
    # 'r109': (react_lower_bound, react_upper_bound),
    # 'r110': (react_lower_bound, react_upper_bound),
    # 'r111': (react_lower_bound, react_upper_bound),
    # 'r112': (react_lower_bound, react_upper_bound),
    # 'r113': (react_lower_bound, react_upper_bound),
    # 'r114': (react_lower_bound, react_upper_bound),
    # 'r115': (react_lower_bound, react_upper_bound),
    # 'r116': (react_lower_bound, react_upper_bound),
    # 'r117': (react_lower_bound, react_upper_bound),
    # 'r118': (react_lower_bound, react_upper_bound),
    # 'r119': (react_lower_bound, react_upper_bound),
    # 'r120': (react_lower_bound, react_upper_bound),
    # 'r121': (react_lower_bound, react_upper_bound),
    # 'r122': (react_lower_bound, react_upper_bound),
    # 'r123': (react_lower_bound, react_upper_bound),
    # 'r124': (react_lower_bound, react_upper_bound),
    # 'r125': (react_lower_bound, react_upper_bound),
    # 'r126': (react_lower_bound, react_upper_bound),
    # 'r127': (react_lower_bound, react_upper_bound),
    # 'r128': (react_lower_bound, react_upper_bound),
    # 'r129': (react_lower_bound, react_upper_bound),
    # 'r130': (react_lower_bound, react_upper_bound),
    # 'r131': (react_lower_bound, react_upper_bound),
    # 'r132': (react_lower_bound, react_upper_bound),
    # 'r133': (react_lower_bound, react_upper_bound),
    # 'r134': (react_lower_bound, react_upper_bound),
    # 'r135': (react_lower_bound, react_upper_bound),
    # 'r136': (react_lower_bound, react_upper_bound),
    # 'r137': (react_lower_bound, react_upper_bound),
    # 'r138': (react_lower_bound, react_upper_bound),
    # 'r139': (react_lower_bound, react_upper_bound),
    # 'r140': (react_lower_bound, react_upper_bound),
    # 'r141': (react_lower_bound, react_upper_bound),
    # 'r142': (react_lower_bound, react_upper_bound),
    # 'r143': (react_lower_bound, react_upper_bound),
    # 'r144': (react_lower_bound, react_upper_bound),
    # 'r145': (react_lower_bound, react_upper_bound),
    # 'r146': (react_lower_bound, react_upper_bound),
    # 'r147': (react_lower_bound, react_upper_bound),
    # 'r148': (react_lower_bound, react_upper_bound),
    # 'r149': (react_lower_bound, react_upper_bound),
    # 'r150': (react_lower_bound, react_upper_bound),
    # 'r151': (react_lower_bound, react_upper_bound),
    # 'r152': (react_lower_bound, react_upper_bound),
    # 'r153': (react_lower_bound, react_upper_bound),
    # 'r154': (react_lower_bound, react_upper_bound),
    # 'r155': (react_lower_bound, react_upper_bound),
    # 'r156': (react_lower_bound, react_upper_bound),
    # 'r157': (react_lower_bound, react_upper_bound),
    # 'r158': (react_lower_bound, react_upper_bound),
    # 'r159': (react_lower_bound, react_upper_bound),
    # 'r160': (react_lower_bound, react_upper_bound),
    # 'r161': (react_lower_bound, react_upper_bound),
    # 'r162': (react_lower_bound, react_upper_bound),
    # 'r163': (react_lower_bound, react_upper_bound),
    # 'r164': (react_lower_bound, react_upper_bound),
    # 'r165': (react_lower_bound, react_upper_bound),
    # 'r166': (react_lower_bound, react_upper_bound),
    # 'r167': (react_lower_bound, react_upper_bound),
    # 'r168': (react_lower_bound, react_upper_bound),
    # 'r169': (react_lower_bound, react_upper_bound),
    # 'r170': (react_lower_bound, react_upper_bound),
    # 'r171': (react_lower_bound, react_upper_bound),
    # 'r172': (react_lower_bound, react_upper_bound),
    # 'r173': (react_lower_bound, react_upper_bound),
    # 'r174': (react_lower_bound, react_upper_bound),
    # 'r175': (react_lower_bound, react_upper_bound),
    # 'r176': (react_lower_bound, react_upper_bound),
    # 'r177': (react_lower_bound, react_upper_bound),
    # 'r178': (react_lower_bound, react_upper_bound),
    # 'r179': (react_lower_bound, react_upper_bound),
    # 'r180': (react_lower_bound, react_upper_bound),
    # 'r181': (react_lower_bound, react_upper_bound),
    # 'r182': (react_lower_bound, react_upper_bound),
    # 'r183': (react_lower_bound, react_upper_bound),
    # 'r184': (react_lower_bound, react_upper_bound),
    # 'r185': (react_lower_bound, react_upper_bound),
    # 'r186': (react_lower_bound, react_upper_bound),
    # 'r187': (react_lower_bound, react_upper_bound),
    # 'r188': (react_lower_bound, react_upper_bound),
    # 'r189': (react_lower_bound, react_upper_bound),
    # 'r190': (react_lower_bound, react_upper_bound),
    # 'r191': (react_lower_bound, react_upper_bound),
    # 'r192': (react_lower_bound, react_upper_bound),
    # 'r193': (react_lower_bound, react_upper_bound),
    # 'r194': (react_lower_bound, react_upper_bound),
    # 'r195': (react_lower_bound, react_upper_bound),
    # 'r196': (react_lower_bound, react_upper_bound),
    # 'r197': (react_lower_bound, react_upper_bound),
    # 'r198': (react_lower_bound, react_upper_bound),
    # 'r199': (react_lower_bound, react_upper_bound),

    'c0': (0, ash_bound),
    'c1': (0, ash_bound),
    'c2': (0, ash_bound),
    'c3': (0, ash_bound),
    'c4': (0, ash_bound),
    'c5': (0, ash_bound),
    'c6': (0, ash_bound),
    'c7': (0, ash_bound),
    'c8': (0, ash_bound),
    'c9': (0, ash_bound),
    'c10': (0, ash_bound),
    'c11': (0, ash_bound),
    'c12': (0, ash_bound),
    'c13': (0, ash_bound),
    'c14': (0, ash_bound),
    'c15': (0, ash_bound),
    'c16': (0, ash_bound),
    'c17': (0, ash_bound),
    'c18': (0, ash_bound),
    'c19': (0, ash_bound),
    'c20': (0, ash_bound),
    'c21': (0, ash_bound),
    'c22': (0, ash_bound),
    'c23': (0, ash_bound),
    'c24': (0, ash_bound),
    'c25': (0, ash_bound),
    'c26': (0, ash_bound),
    'c27': (0, ash_bound),
    'c28': (0, ash_bound),
    'c29': (0, ash_bound),
    'c30': (0, ash_bound),
    'c31': (0, ash_bound),
    'c32': (0, ash_bound),
    'c33': (0, ash_bound),
    'c34': (0, ash_bound),
    'c35': (0, ash_bound),
    'c36': (0, ash_bound),
    'c37': (0, ash_bound),
    'c38': (0, ash_bound),
    'c39': (0, ash_bound),
    'c40': (0, ash_bound),
    'c41': (0, ash_bound),
    'c42': (0, ash_bound),
    'c43': (0, ash_bound),
    'c44': (0, ash_bound),
    'c45': (0, ash_bound),
    'c46': (0, ash_bound),
    'c47': (0, ash_bound),
    'c48': (0, ash_bound),
    'c49': (0, ash_bound),
    'c50': (0, ash_bound),
    'c51': (0, ash_bound),
    'c52': (0, ash_bound),
    'c53': (0, ash_bound),
    'c54': (0, ash_bound),
    'c55': (0, ash_bound),
    'c56': (0, ash_bound),
    'c57': (0, ash_bound),
    'c58': (0, ash_bound),
    'c59': (0, ash_bound),
    'c60': (0, ash_bound),
    'c61': (0, ash_bound),
    'c62': (0, ash_bound),
    'c63': (0, ash_bound),
    'c64': (0, ash_bound),
    'c65': (0, ash_bound),
    'c66': (0, ash_bound),
    'c67': (0, ash_bound),
    'c68': (0, ash_bound),
    'c69': (0, ash_bound),
    'c70': (0, ash_bound),
    'c71': (0, ash_bound),
    'c72': (0, ash_bound),
    'c73': (0, ash_bound),
    'c74': (0, ash_bound),
    'c75': (0, ash_bound),
    'c76': (0, ash_bound),
    'c77': (0, ash_bound),
    'c78': (0, ash_bound),
    'c79': (0, ash_bound),
    # 'c80': (0, ash_bound),
    # 'c81': (0, ash_bound),
    # 'c82': (0, ash_bound),
    # 'c83': (0, ash_bound),
    # 'c84': (0, ash_bound),
    # 'c85': (0, ash_bound),
    # 'c86': (0, ash_bound),
    # 'c87': (0, ash_bound),
    # 'c88': (0, ash_bound),
    # 'c89': (0, ash_bound),
    # 'c90': (0, ash_bound),
    # 'c91': (0, ash_bound),
    # 'c92': (0, ash_bound),
    # 'c93': (0, ash_bound),
    # 'c94': (0, ash_bound),
    # 'c95': (0, ash_bound),
    # 'c96': (0, ash_bound),
    # 'c97': (0, ash_bound),
    # 'c98': (0, ash_bound),
    # 'c99': (0, ash_bound),
    # 'c100': (0, ash_bound),
    # 'c101': (0, ash_bound),
    # 'c102': (0, ash_bound),
    # 'c103': (0, ash_bound),
    # 'c104': (0, ash_bound),
    # 'c105': (0, ash_bound),
    # 'c106': (0, ash_bound),
    # 'c107': (0, ash_bound),
    # 'c108': (0, ash_bound),
    # 'c109': (0, ash_bound),
    # 'c110': (0, ash_bound),
    # 'c111': (0, ash_bound),
    # 'c112': (0, ash_bound),
    # 'c113': (0, ash_bound),
    # 'c114': (0, ash_bound),
    # 'c115': (0, ash_bound),
    # 'c116': (0, ash_bound),
    # 'c117': (0, ash_bound),
    # 'c118': (0, ash_bound),
    # 'c119': (0, ash_bound),
    # 'c120': (0, ash_bound),
    # 'c121': (0, ash_bound),
    # 'c122': (0, ash_bound),
    # 'c123': (0, ash_bound),
    # 'c124': (0, ash_bound),
    # 'c125': (0, ash_bound),
    # 'c126': (0, ash_bound),
    # 'c127': (0, ash_bound),
    # 'c128': (0, ash_bound),
    # 'c129': (0, ash_bound),
    # 'c130': (0, ash_bound),
    # 'c131': (0, ash_bound),
    # 'c132': (0, ash_bound),
    # 'c133': (0, ash_bound),
    # 'c134': (0, ash_bound),
    # 'c135': (0, ash_bound),
    # 'c136': (0, ash_bound),
    # 'c137': (0, ash_bound),
    # 'c138': (0, ash_bound),
    # 'c139': (0, ash_bound),
    # 'c140': (0, ash_bound),
    # 'c141': (0, ash_bound),
    # 'c142': (0, ash_bound),
    # 'c143': (0, ash_bound),
    # 'c144': (0, ash_bound),
    # 'c145': (0, ash_bound),
    # 'c146': (0, ash_bound),
    # 'c147': (0, ash_bound),
    # 'c148': (0, ash_bound),
    # 'c149': (0, ash_bound),
    # 'c150': (0, ash_bound),
    # 'c151': (0, ash_bound),
    # 'c152': (0, ash_bound),
    # 'c153': (0, ash_bound),
    # 'c154': (0, ash_bound),
    # 'c155': (0, ash_bound),
    # 'c156': (0, ash_bound),
    # 'c157': (0, ash_bound),
    # 'c158': (0, ash_bound),
    # 'c159': (0, ash_bound),
    # 'c160': (0, ash_bound),
    # 'c161': (0, ash_bound),
    # 'c162': (0, ash_bound),
    # 'c163': (0, ash_bound),
    # 'c164': (0, ash_bound),
    # 'c165': (0, ash_bound),
    # 'c166': (0, ash_bound),
    # 'c167': (0, ash_bound),
    # 'c168': (0, ash_bound),
    # 'c169': (0, ash_bound),
    # 'c170': (0, ash_bound),
    # 'c171': (0, ash_bound),
    # 'c172': (0, ash_bound),
    # 'c173': (0, ash_bound),
    # 'c174': (0, ash_bound),
    # 'c175': (0, ash_bound),
    # 'c176': (0, ash_bound),
    # 'c177': (0, ash_bound),
    # 'c178': (0, ash_bound),
    # 'c179': (0, ash_bound),
    # 'c180': (0, ash_bound),
    # 'c181': (0, ash_bound),
    # 'c182': (0, ash_bound),
    # 'c183': (0, ash_bound),
    # 'c184': (0, ash_bound),
    # 'c185': (0, ash_bound),
    # 'c186': (0, ash_bound),
    # 'c187': (0, ash_bound),
    # 'c188': (0, ash_bound),
    # 'c189': (0, ash_bound),
    # 'c190': (0, ash_bound),
    # 'c191': (0, ash_bound),
    # 'c192': (0, ash_bound),
    # 'c193': (0, ash_bound),
    # 'c194': (0, ash_bound),
    # 'c195': (0, ash_bound),
    # 'c196': (0, ash_bound),
    # 'c197': (0, ash_bound),
    # 'c198': (0, ash_bound),
    # 'c199': (0, ash_bound),
    },
    allow_duplicate_points=True,
    random_state=args.seed,
)

if args.acquisition == "ucb":
    acquisition_function = UtilityFunction(kind="ucb", kappa=args.kappa)
elif args.acquisition == "ei":
    acquisition_function = UtilityFunction(kind="ei", xi=args.xi)
elif args.acquisition == "poi":
    acquisition_function = UtilityFunction(kind="poi", xi=args.xi)

ood_bayesian.maximize(
    init_points=50,
    n_iter=450,
    acquisition_function=acquisition_function,
)

# todo
def feats_to_score(feats, loader):
    global id_feats
    extract_feats(feats, loader)
    feats = torch.cat(feats, dim=0)
    error = feats - torch.Tensor(nmf.transform(feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    trans = torch.clone(torch.Tensor(nmf.components_))

    feats = torch.Tensor(nmf.transform(feats.cpu())).mm(trans.T).cuda() + error

    if args.model == "resnet50" or args.model == "densenet_dice" or args.model == "mobilenet" or args.model == "wrn":
        logits = model.fc(feats)
    elif args.model == "wideresnet":
        logits = model.linear(feats)
    score =  - torch.logsumexp(logits, axis=1).cpu().detach().numpy()
    return score

def evaluate():
    print(id_feats_eval)
    id_score_eval = feats_to_score(id_feats_eval, id_loader_eval)
    if torch.isnan(torch.Tensor(id_score_eval)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(id_score_eval.shape)
    print(texture_feats)
    texture_score = feats_to_score(texture_feats, texture_loader)
    if torch.isnan(torch.Tensor(texture_score)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(texture_score.shape)
    print(places365_feats)
    places365_score = feats_to_score(places365_feats, places365_loader)
    if torch.isnan(torch.Tensor(places365_score)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(places365_score.shape)
    print(lsunc_feats)
    lsunc_score = feats_to_score(lsunc_feats, lsunc_loader)
    if torch.isnan(torch.Tensor(lsunc_score)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(lsunc_score.shape)
    print(lsunr_feats)
    lsunr_score = feats_to_score(lsunr_feats, lsunr_loader)
    if torch.isnan(torch.Tensor(lsunr_score)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(lsunr_score.shape)
    print(isun_feats)
    isun_score = feats_to_score(isun_feats, isun_loader)
    if torch.isnan(torch.Tensor(isun_score)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(isun_score.shape)
    print(svhn_feats)
    svhn_score = feats_to_score(svhn_feats, svhn_loader)
    if torch.isnan(torch.Tensor(svhn_score)).sum() != 0:
        raise ValueError("The tensor has nan or inf elements")
    print(svhn_score.shape)
    texture_fpr, _, _ = score_get_and_print_results(log, id_score_eval, texture_score)
    places365_fpr, _, _ = score_get_and_print_results(log, id_score_eval, places365_score)
    lsunc_fpr, _, _ = score_get_and_print_results(log, id_score_eval, lsunc_score)
    lsunr_fpr, _, _ = score_get_and_print_results(log, id_score_eval, lsunr_score)
    isun_fpr, _, _ = score_get_and_print_results(log, id_score_eval, isun_score)
    svhn_fpr, _, _ = score_get_and_print_results(log, id_score_eval, svhn_score)
    print("avg_fpr: %.2f" % ((texture_fpr + places365_fpr + lsunc_fpr + lsunr_fpr + isun_fpr + svhn_fpr) / 6 * 100))
    print("avg_fpr_wo_texture: %.2f" % ((places365_fpr + lsunc_fpr + lsunr_fpr + isun_fpr + svhn_fpr) / 5 * 100))
evaluate()
