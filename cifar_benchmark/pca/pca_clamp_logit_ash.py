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

import models.hybrid_resnet as resnet
from models.densenet import densenet161
from models.resnet_cifar_ash import ResNet34, ResNet50
from models.densenet_dice import DenseNet3
# from models.densenet_ash import DenseNet3
from utils.svhn_loader import SVHN
import numpy as np
import sklearn.metrics as sk
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import NMF, FastICA
from bayes_opt import BayesianOptimization

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet161", "densenet_dice", "densenet_ash"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--use_react", type=str, choices=["yes", "no"])
parser.add_argument("--variance", type=str, choices=["only_react", "ash_p", "ash_b", "ash_s"])
parser.add_argument("--train_trans", type=str, choices=["id", "train"])
parser.add_argument("--auxiliary_trans", type=str, choices=["ood", "eval"])
parser.add_argument("--seq_choice", type=str, choices=["react_first", "ash_first"])
parser.add_argument("--num_component", type=int)
parser.add_argument("--method", type=str, choices=["pca", "skpca", "nmf", "ica", "tsne"])
parser.add_argument("--component_normalization", type=str, choices=["yes", "no"])
parser.add_argument("--ash_choice", type=str, choices=["percent", "global_percent", "threshold"])
# parser.add_argument("--percent", type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.95, 0.99])
args = parser.parse_args()

recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()
id_logits = []
id_logits_test = []
ood_logits = []
texture_logits = []
places365_logits = []
lsunc_logits = []
lsunr_logits = []
isun_logits = []
svhn_logits = []

# torch.cuda.set_device(2)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

log = logging.getLogger("hybrid")

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
    model = resnet.resnet50(num_classes=10)
    model.load_state_dict(torch.load("./ckpt/resnet50_cifar10_0.9501.pt"))
elif args.dataset == "cifar100":
    if args.train_trans == "train":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
    elif args.train_trans == "id":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=id_transform, download=False)
    id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)

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

# ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))
# texture_data = dset.ImageFolder(root="../data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
# places365_data = dset.ImageFolder(root="../data/places365", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
# lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
# lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
# isun_data = dset.ImageFolder(root="../data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

ood_transform=trn.Compose([
    trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
    trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)
])

eval_transform = trn.Compose([
    trn.Resize(32),
    trn.CenterCrop(32),
    trn.ToTensor(),
    # trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trn.Normalize(mean, std)
])

# CUDA out of memory
# eval_transform = trn.Compose([
#     trn.Resize(256),
#     trn.CenterCrop(224),
#     trn.ToTensor(),
#     trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

if args.auxiliary_trans == "ood":
    # ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
    ood_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform) # 修改
elif args.auxiliary_trans == "eval":
    # ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=eval_transform)
    ood_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform) # 修改
texture_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
places365_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
isun_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
svhn_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

id_loader = torch.utils.data.DataLoader(id_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

def ash_p(x, percent):
    k = x.shape[1] - int(np.round(x.shape[1] * percent))
    v, i = torch.topk(x, k, dim=1)
    # v, i = torch.topk(x, k, dim=1, largest=False)
    x.zero_().scatter_(dim=1, index=i, src=v)
    return x

def ash_b(x, percent):
    s1 = x.sum(dim=1)
    k = x.shape[1] - int(np.round(x.shape[1] * percent))
    v, i = torch.topk(x, k, dim=1)
    # v, i = torch.topk(t, k, dim=1, largest=False)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    x.zero_().scatter_(dim=1, index=i, src=fill)
    return x

def ash_s(x, percent):
# def ash_s(x, percent, sca): # scale
    s1 = x.sum(dim=1)
    k = x.shape[1] - int(np.round(x.shape[1] * percent))
    v, i = torch.topk(x, k, dim=1)
    # v, i = torch.topk(x, k, dim=1, largest=False)
    x.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=1)
    scale = s1 / s2
    x *= torch.exp(scale[:, None])
    # x *= torch.exp(scale[:, None] * sca) # scale
    return x

def ash_p_n(x, percent):
    k = x.shape[1] - int(np.round(x.shape[1] * percent))
    # v, i = torch.topk(x, k, dim=1)
    v, i = torch.topk(x, k, dim=1, largest=False)
    x.zero_().scatter_(dim=1, index=i, src=v)
    return x

def ash_b_n(x, percent):
    s1 = x.sum(dim=1)
    k = x.shape[1] - int(np.round(x.shape[1] * percent))
    # v, i = torch.topk(x, k, dim=1)
    v, i = torch.topk(x, k, dim=1, largest=False)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    x.zero_().scatter_(dim=1, index=i, src=fill)
    return x

# def ash_s_n(x, percent):
def ash_s_n(x, percent, sca): # scale
    s1 = x.sum(dim=1)
    k = x.shape[1] - int(np.round(x.shape[1] * percent))
    # v, i = torch.topk(x, k, dim=1)
    v, i = torch.topk(x, k, dim=1, largest=False)
    x.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=1)
    scale = s1 / s2
    # x *= torch.exp(scale[:, None])
    x *= torch.exp(scale[:, None] * sca) # scale
    return x

def ash_p_thre(x, threshold):
    k = (x >= threshold).sum()
    # k = (x > threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    # v, i = torch.topk(t, k, dim=1, largest=False)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x

def ash_b_thre(x, threshold):
    s1 = x.sum()
    k = (x >= threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x

# def ash_s_thre(x, threshold):
def ash_s_thre(x, threshold, sca): # scale
    s1 = x.sum()
    k = (x >= threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum()
    scale = s1 / s2
    # x *= torch.exp(scale)
    x *= torch.exp(scale * sca) # scale
    return x

def ash_p_global(x, percent):
    k = x.shape[0] * x.shape[1] - int(np.round(x.shape[0] * x.shape[1] * percent))
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    # v, i = torch.topk(t, k, dim=1, largest=False)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x

def ash_b_global(x, percent):
    s1 = x.sum()
    k = x.shape[0] * x.shape[1] - int(np.round(x.shape[0] * x.shape[1] * percent))
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    # v, i = torch.topk(t, k, dim=1, largest=False)
    fill = s1 / k
    fill = torch.full(v.shape, fill)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x

# def ash_s(x, percent):
def ash_s_global(x, percent, sca): # scale
    s1 = x.sum(dim=1)
    k = x.shape[0] * x.shape[1] - int(np.round(x.shape[0] * x.shape[1] * percent))
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    # v, i = torch.topk(t, k, dim=1, largest=False)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=1)
    scale = s1 / s2
    # x *= torch.exp(scale[:, None])
    x *= torch.exp(scale * sca) # scale
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

def extract_logits(logits, loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            logits.append(model.(data))

acc = test(id_loader_test)
print("acc: %.4f" % (acc))

extract_logits(id_logits, id_loader)
extract_logits(ood_logits, ood_loader)
id_logits = torch.cat(id_logits, dim=0)
ood_logits = torch.cat(ood_logits, dim=0)

if args.method == "pca":
    pca = PCA(n_components=args.num_component)
    pca.fit(id_logits)
elif args.method == "skpca":
    skpca = skPCA(n_components=args.num_component)
    skpca.fit(id_logits.cpu())
elif args.method == "ica":
    ica = FastICA(n_components=args.num_component)
    ica.fit(id_logits.cpu())
elif args.method == "nmf":
    nmf = NMF(n_components=args.num_component)
    nmf.fit(id_logits.cpu())

def eval_datasets(clamp1_1, clamp1_2, clamp1_3):
    global id_logits, ood_logits
    m_id_logits = torch.clone(id_logits)
    m_ood_logits = torch.clone(ood_logits)

    if args.method == "pca":
        m_id_error = m_id_logits - pca.transform(m_id_logits).mm(pca.proj.T)
        m_ood_error = m_ood_logits - pca.transform(m_ood_logits).mm(pca.proj.T)
        m_trans = torch.clone(pca.proj)
    elif args.method == "skpca":
        # m_id_error = m_id_logits - (torch.Tensor(skpca.transform(m_id_logits.cpu())).mm(torch.Tensor(skpca.components_)).cuda() + m_id_logits.mean(0))
        # m_ood_error = m_ood_logits - (torch.Tensor(skpca.transform(m_ood_logits.cpu())).mm(torch.Tensor(skpca.components_)).cuda() + m_ood_logits.mean(0))
        m_id_error = m_id_logits - torch.Tensor(skpca.inverse_transform(skpca.transform(m_id_logits.cpu()))).cuda() # 修改
        m_ood_error = m_ood_logits - torch.Tensor(skpca.inverse_transform(skpca.transform(m_ood_logits.cpu()))).cuda() # 修改
        m_trans = torch.clone(torch.Tensor(skpca.components_.T))
    elif args.method == "ica":
        m_id_error = m_id_logits - (torch.Tensor(ica.transform(m_id_logits.cpu())).mm(torch.pinverse(torch.Tensor(ica.components_.T))).cuda() + m_id_logits.mean(0))
        m_ood_error = m_ood_logits - (torch.Tensor(ica.transform(m_ood_logits.cpu())).mm(torch.pinverse(torch.Tensor(ica.components_.T))).cuda() + m_ood_logits.mean(0))
        m_trans = torch.clone(torch.Tensor(ica.components_.T))
    elif args.method == "nmf":
        m_id_error = m_id_logits - torch.Tensor(nmf.transform(m_id_logits.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood_error = m_ood_logits - torch.Tensor(nmf.transform(m_ood_logits.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_trans = torch.clone(torch.Tensor(nmf.components_.T))

    if args.seq_choice == "react_first":
        print("react_first")
        if args.use_react == "yes":
            m_trans = react(m_trans, clamp1_1)
        if args.variance == "ash_p":
            if args.ash_choice == "percent":
                if clamp1_2 >= 0:
                    m_trans = ash_p(m_trans, clamp1_2)
                elif clamp1_2 < 0:
                    m_trans = ash_p_n(m_trans, -clamp1_2)
            elif args.ash_choice == "global_percent":
                m_trans = ash_p_global(m_trans, clamp1_2)
            elif args.ash_choice == "threshold":
                m_trans = ash_p_thre(m_trans, clamp1_2)
        elif args.variance == "ash_b": 
            if args.ash_choice == "percent":
                if clamp1_2 >= 0:
                    m_trans = ash_b(m_trans, clamp1_2)
                elif clamp1_2 < 0:
                    m_trans = ash_b_n(m_trans, -clamp1_2)
            elif args.ash_choice == "global_percent":
                m_trans = ash_b_global(m_trans, clamp1_2)
            elif args.ash_choice == "threshold":
                m_trans = ash_b_thre(m_trans, clamp1_2)
        elif args.variance == "ash_s":
            if args.ash_choice == "percent":
                if clamp1_2 >= 0:
                    m_trans = ash_s(m_trans, clamp1_2, clamp1_3)
                elif clamp1_2 < 0:
                    m_trans = ash_s_n(m_trans, -clamp1_2, clamp1_3)
            elif args.ash_choice == "global_percent":
                m_trans = ash_s_global(m_trans, clamp1_2, clamp1_3)
            elif args.ash_choice == "threshold":
                m_trans = ash_s_thre(m_trans, clamp1_2, clamp1_3)
    elif args.seq_choice == "ash_first":
        print("ash_first")
        if args.variance == "ash_p":
            if args.ash_choice == "percent":
                if clamp1_2 >= 0:
                    m_trans = ash_p(m_trans, clamp1_2)
                elif clamp1_2 < 0:
                    m_trans = ash_p_n(m_trans, -clamp1_2)
            elif args.ash_choice == "global_percent":
                m_trans = ash_p_global(m_trans, clamp1_2)
            elif args.ash_choice == "threshold":
                m_trans = ash_p_thre(m_trans, clamp1_2)
        elif args.variance == "ash_b": 
            if args.ash_choice == "percent":
                if clamp1_2 >= 0:
                    m_trans = ash_b(m_trans, clamp1_2)
                elif clamp1_2 < 0:
                    m_trans = ash_b_n(m_trans, -clamp1_2)
            elif args.ash_choice == "global_percent":
                m_trans = ash_b_global(m_trans, clamp1_2)
            elif args.ash_choice == "threshold":
                m_trans = ash_b_thre(m_trans, clamp1_2)
        elif args.variance == "ash_s":
            if args.ash_choice == "percent":
                if clamp1_2 >= 0:
                    m_trans = ash_s(m_trans, clamp1_2, clamp1_3)
                elif clamp1_2 < 0:
                    m_trans = ash_s_n(m_trans, -clamp1_2, clamp1_3)
            elif args.ash_choice == "global_percent":
                m_trans = ash_s_global(m_trans, clamp1_2, clamp1_3)
            elif args.ash_choice == "threshold":
                m_trans = ash_s_thre(m_trans, clamp1_2, clamp1_3)
        if args.use_react == "yes":
            m_trans = react(m_trans, clamp1_1)

    if args.component_normalization == "yes":
        m_trans = m_trans / torch.norm(m_trans)

    if args.method == "pca":
        m_id_logits = pca.transform(m_id_logits).mm(m_trans.T) + m_id_error
        m_ood_logits = pca.transform(m_ood_logits).mm(m_trans.T) + m_ood_error
    elif args.method == "skpca":
        m_id_logits = torch.Tensor(skpca.transform(m_id_logits.cpu())).mm(m_trans.T).cuda() + m_id_logits.mean(0) + m_id_error
        m_ood_logits = torch.Tensor(skpca.transform(m_ood_logits.cpu())).mm(m_trans.T).cuda() + m_ood_logits.mean(0) + m_ood_error
    elif args.method == "ica":
        m_id_logits = torch.Tensor(ica.transform(m_id_logits.cpu())).mm(torch.pinverse(m_trans)).cuda() + m_id_logits.mean(0) + m_id_error
        m_ood_logits = torch.Tensor(ica.transform(m_ood_logits.cpu())).mm(torch.pinverse(m_trans)).cuda() + m_ood_logits.mean(0) + m_ood_error
    elif args.method == "nmf":
        m_id_logits = torch.Tensor(nmf.transform(m_id_logits.cpu())).mm(m_trans.T).cuda() + m_id_error
        m_ood_logits = torch.Tensor(nmf.transform(m_ood_logits.cpu())).mm(m_trans.T).cuda() + m_ood_error

    if args.model == "densenet_dice" or args.model == "densenet_ash":
        m_id_logits = model.fc(m_id_logits)
        m_ood_logits = model.fc(m_ood_logits)
    elif args.model == "densenet161":
        m_id_logits = model.linear(m_id_logits)
        m_ood_logits = model.linear(m_ood_logits)
    m_id_score =  - torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
    m_ood_score =  - torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()
    # m_ood_score = np.sort(m_ood_score)[:20000]
    # m_ood_score = np.sort(m_ood_score)[20000:80000]
    # m_ood_score = np.sort(m_ood_score)[80000:]
    
    fpr, auroc, aupr = score_get_and_print_results(log, m_id_score, m_ood_score)
    # return auroc + aupr - fpr
    # return auroc + aupr - fpr + (clamp1_2 / 10)
    # return auroc + aupr - fpr + (clamp1_2 / 5)
    # return auroc + aupr - fpr - (clamp1_2 / 10)
    return auroc + aupr - fpr

ood_bayesian = BayesianOptimization(
    eval_datasets,
    {
    'clamp1_1': (0, 10),
    'clamp1_2': (0, 10),
    # 'clamp1_2': (0, 1),
    'clamp1_3': (0.01, 10)
    },
    allow_duplicate_points=True,
)

ood_bayesian.maximize(
    init_points=30,
    n_iter=30,
)

def logits_to_score(logits, loader):
    extract_logits(logits, loader)
    logits = torch.cat(logits, dim=0)
    if args.method == "pca":
        error = logits - pca.transform(logits).mm(pca.proj.T)
        trans = torch.clone(pca.proj)
    elif args.method == "skpca":
        # error = logits - (torch.Tensor(skpca.transform(logits.cpu())).mm(torch.Tensor(skpca.components_)).cuda() + logits.mean(0))
        error = logits - torch.Tensor(skpca.inverse_transform(skpca.transform(logits.cpu()))).cuda() # 修改
        trans = torch.clone(torch.Tensor(skpca.components_.T))
    elif args.method == "ica":
        error = logits - (torch.Tensor(ica.transform(logits.cpu())).mm(torch.pinverse(torch.Tensor(ica.components_.T))).cuda() + logits.mean(0))
        trans = torch.clone(torch.Tensor(ica.components_.T))
    elif args.method == "nmf":
        error = logits - torch.Tensor(nmf.transform(logits.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        trans = torch.clone(torch.Tensor(nmf.components_.T))

    if args.seq_choice == "react_first":
        if args.use_react == "yes":
            trans = react(trans, ood_bayesian.max["params"]["clamp1_1"])

        if args.variance == "ash_p":
            if args.ash_choice == "percent":
                if ood_bayesian.max["params"]["clamp1_2"] >= 0:
                    trans = ash_p(trans, ood_bayesian.max["params"]["clamp1_2"])
                elif ood_bayesian.max["params"]["clamp1_2"] < 0:
                    trans = ash_p_n(trans, -ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "global_percent":
                trans = ash_p_global(trans, ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "threshold":
                trans = ash_p_thre(trans, ood_bayesian.max["params"]["clamp1_2"])
            # trans = ash_p(trans, 0.91)
        if args.variance == "ash_b":
            if args.ash_choice == "percent":
                if ood_bayesian.max["params"]["clamp1_2"] >= 0:
                    trans = ash_b(trans, ood_bayesian.max["params"]["clamp1_2"])
                elif ood_bayesian.max["params"]["clamp1_2"] < 0:
                    trans = ash_b_n(trans, -ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "global_percent":
                trans = ash_b_global(trans, ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "threshold":
                trans = ash_b_thre(trans, ood_bayesian.max["params"]["clamp1_2"])
            # trans = ash_b(trans, 0.91)
        if args.variance == "ash_s":
            if args.ash_choice == "percent":
                if ood_bayesian.max["params"]["clamp1_2"] >= 0:
                    trans = ash_s(trans, ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
                elif ood_bayesian.max["params"]["clamp1_2"] < 0:
                    trans = ash_s_n(trans, -ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            elif args.ash_choice == "global_percent":
                trans = ash_s_global(trans, ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            elif args.ash_choice == "threshold":
                trans = ash_s_thre(trans, ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            # trans = ash_s(trans, 0.91)
    elif args.seq_choice == "ash_first":
        if args.variance == "ash_p":
            if args.ash_choice == "percent":
                if ood_bayesian.max["params"]["clamp1_2"] >= 0:
                    trans = ash_p(trans, ood_bayesian.max["params"]["clamp1_2"])
                elif ood_bayesian.max["params"]["clamp1_2"] < 0:
                    trans = ash_p_n(trans, -ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "global_percent":
                trans = ash_p_global(trans, ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "threshold":
                trans = ash_p_thre(trans, ood_bayesian.max["params"]["clamp1_2"])
            # trans = ash_p(trans, 0.91)
        if args.variance == "ash_b":
            if args.ash_choice == "percent":
                if ood_bayesian.max["params"]["clamp1_2"] >= 0:
                    trans = ash_b(trans, ood_bayesian.max["params"]["clamp1_2"])
                elif ood_bayesian.max["params"]["clamp1_2"] < 0:
                    trans = ash_b_n(trans, -ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "global_percent":
                trans = ash_b_global(trans, ood_bayesian.max["params"]["clamp1_2"])
            elif args.ash_choice == "threshold":
                trans = ash_b_thre(trans, ood_bayesian.max["params"]["clamp1_2"])
            # trans = ash_b(trans, 0.91)
        if args.variance == "ash_s":
            if args.ash_choice == "percent":
                if ood_bayesian.max["params"]["clamp1_2"] >= 0:
                    trans = ash_s(trans, ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
                elif ood_bayesian.max["params"]["clamp1_2"] < 0:
                    trans = ash_s_n(trans, -ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            elif args.ash_choice == "global_percent":
                trans = ash_s_global(trans, ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            elif args.ash_choice == "threshold":
                trans = ash_s_thre(trans, ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            # trans = ash_s(trans, 0.91)

        if args.use_react == "yes":
            trans = react(trans, ood_bayesian.max["params"]["clamp1_1"])

    if args.component_normalization == "yes":
        trans = trans / torch.norm(trans)

    if args.method == "pca":
        logits = pca.transform(logits).mm(trans.T) + error
    elif args.method == "skpca":
        logits = torch.Tensor(skpca.transform(logits.cpu())).mm(trans.T).cuda() + logits.mean(0) + error
    elif args.method == "ica":
        logits = torch.Tensor(ica.transform(logits.cpu())).mm(torch.pinverse(trans)).cuda() + logits.mean(0) + error
    elif args.method == "nmf":
        logits = torch.Tensor(nmf.transform(logits.cpu())).mm(trans.T).cuda() + error

    score =  - torch.logsumexp(logits, axis=1).cpu().detach().numpy()
    return score

def evaluate():
    print(id_logits_test)
    id_score_test = logits_to_score(id_logits_test, id_loader_test)
    print(id_score_test.shape)
    print(texture_logits)
    texture_score = logits_to_score(texture_logits, texture_loader)
    print(texture_score.shape)
    print(places365_logits)
    places365_score = logits_to_score(places365_logits, places365_loader)
    print(places365_score.shape)
    print(lsunc_logits)
    lsunc_score = logits_to_score(lsunc_logits, lsunc_loader)
    print(lsunc_score.shape)
    print(lsunr_logits)
    lsunr_score = logits_to_score(lsunr_logits, lsunr_loader)
    print(lsunr_score.shape)
    print(isun_logits)
    isun_score = logits_to_score(isun_logits, isun_loader)
    print(isun_score.shape)
    print(svhn_logits)
    svhn_score = logits_to_score(svhn_logits, svhn_loader)
    print(svhn_score.shape)
    # ood_fpr, _, _ = score_get_and_print_results(log, id_score, ood_score)
    texture_fpr, _, _ = score_get_and_print_results(log, id_score_test, texture_score)
    places365_fpr, _, _ = score_get_and_print_results(log, id_score_test, places365_score)
    lsunc_fpr, _, _ = score_get_and_print_results(log, id_score_test, lsunc_score)
    lsunr_fpr, _, _ = score_get_and_print_results(log, id_score_test, lsunr_score)
    isun_fpr, _, _ = score_get_and_print_results(log, id_score_test, isun_score)
    svhn_fpr, _, _ = score_get_and_print_results(log, id_score_test, svhn_score)
    print("avg_fpr: %.2f" % ((texture_fpr + places365_fpr + lsunc_fpr + lsunr_fpr + isun_fpr + svhn_fpr) / 6 * 100))
evaluate()
