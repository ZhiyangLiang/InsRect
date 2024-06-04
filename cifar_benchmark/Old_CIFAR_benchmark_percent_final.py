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
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import NMF, FastICA
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from utils.tinyimages_300K_random_loader import TinyImages

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet161", "densenet_dice", "densenet_ash", "wideresnet", "mobilenet"])
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train_trans", type=str, choices=["id", "train"])
parser.add_argument("--auxiliary_trans", type=str, choices=["ood", "eval"])
parser.add_argument("--num_component", type=int)
parser.add_argument("--method", type=str, choices=["pca", "skpca", "nmf", "ica"])
parser.add_argument("--acquisition", type=str, choices=["ucb", "ei", "poi"])
parser.add_argument("--kappa", type=float)
parser.add_argument("--xi", type=float)
parser.add_argument("--use_ash", type=str, choices=["yes", "no"])
parser.add_argument("--react_percent", type=str, choices=["yes", "no"])
parser.add_argument("--lower_percentile", type=float)
parser.add_argument("--upper_percentile", type=float)
parser.add_argument("--ash_percentile", type=float)
parser.add_argument("--ash_scale", type=str, choices=["yes", "no"])
parser.add_argument("--softmax_temperature", type=str, choices=["yes", "no"])
parser.add_argument("--scale", type=str, choices=["yes", "no"])
parser.add_argument("--feat_noise", type=str, choices=["gaussian", "uniform", "none"])
parser.add_argument("--input_noise", type=str, choices=["yes", "no"])
parser.add_argument("--feat_margin", type=str, choices=["one", "two", "fro", "none"])
parser.add_argument("--margin_scale", type=float)
parser.add_argument("--metric", type=str, choices=["fpr", "auroc", "both"])
args = parser.parse_args()

final_fpr = 1.0
final_avg_fpr = 1.0
final_fpr1 = 1.0
final_fpr2 = 1.0
final_fpr3 = 1.0
final_fpr4 = 1.0
final_fpr5 = 1.0
final_fpr6 = 1.0
final_auroc = -1.0
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
elif args.dataset == "cifar100":
    if args.train_trans == "train":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
    elif args.train_trans == "id":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=id_transform, download=False)
    id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)

    if args.model == "resnet50":
        model = resnet.resnet50(num_classes=100)
        model.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
    elif args.model == "densenet161":
        model = densenet161()
        model.load_state_dict(torch.load("./ckpt/densenet161_epoch175_acc0.8011999726295471.pt"))
    elif args.model == "densenet_dice":
        model = DenseNet3(100, 100)
        model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
    elif args.model == "densenet_ash":
        model = DenseNet3(100, 100)
        model.load_state_dict(torch.load("./ckpt/densenet161_ash_cifar100_epoch136_acc0.7649999856948853.pt"))
    elif args.model == "wideresnet":
        model = wideresnet(num_classes=100)
        model.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
    elif args.model == "mobilenet":
        model = mobilenet(class_num=100)
        model.load_state_dict(torch.load("./ckpt/mobilenet_epoch124_acc0.677299976348877.pt", map_location='cuda:0'))
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

class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, feats):
        n = feats.shape[0]
        self.mean = torch.mean(feats, axis=0)
        feats = feats - self.mean
        covariance_matrix = 1 / n * torch.matmul(feats.T, feats)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        eigenvalues = torch.abs(eigenvalues)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj = eigenvectors[:, 0:self.n_components].real

    def transform(self, feats):
        return (feats - self.mean).mm(self.proj)

# def react(x, threshold): # (react_percent)
#     x = torch.clip(x, max=threshold)
#     return x

def react(x, percent):
    threshold = np.percentile(x, percent)
    x = torch.clip(x, max=threshold)
    return x

# def ash_s_thre(x, threshold):
#     s1 = x.sum()
#     k = (x >= threshold).sum()
#     t = x.view((1, -1))
#     v, i = torch.topk(t, k, dim=1)
#     t.zero_().scatter_(dim=1, index=i, src=v)
#     s2 = x.sum()
#     scale = s1 / s2
#     x *= torch.exp(scale)
#     return x

# def ash_s_thre(x, percentile): # (ash_s_percent_set)
#     s1 = x.sum(dim=1)
#     n = x.shape[1]
#     k = n - int(np.round(n * percentile / 100.0))
#     t = x
#     v, i = torch.topk(t, k, dim=1)
#     t.zero_().scatter_(dim=1, index=i, src=v)
#     s2 = x.sum(dim=[1])
#     scale = s1 / s2
#     x = x * torch.exp(scale[:, None])
#     return x

def ash_s_thre(x, percentile): # (ash_s_percent_instance)
    threshold = np.percentile(x.cpu(), percentile)
    # s1 = x.sum()
    k = (x >= threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    # s2 = x.sum()
    # scale = s1 / s2
    # x *= torch.exp(scale)
    return x

def ash_s(x, percentile):
    s1 = x.sum(dim=1)
    n = x.shape[1]
    k = int(n * percentile / 100)
    t = x
    v, i = torch.topk(t, k, dim=1)
    x_relu = nmf_relu(x)
    s2 = x_relu.sum(dim=1)
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

def extract_feats(feats, loader, opt=0):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            if opt == 1:
                data = data + torch.empty_like(data).normal_(0, 0.005) # Gaussian Noise
            feats.append(model.get_features_fc(data))

acc = test(id_loader_test)
print("acc: %.4f" % (acc))
nmf_relu = nn.ReLU(inplace=True)

extract_feats(id_feats, id_loader)
extract_feats(id_feats_test, id_loader_test)
if args.input_noise == "yes":
    extract_feats(ood_feats, ood_loader, 1) # Gaussian Noise
elif args.input_noise == "no":
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

if args.method == "pca":
    pca = PCA(n_components=args.num_component)
    pca.fit(id_feats)
elif args.method == "skpca":
    skpca = skPCA(n_components=args.num_component)
    skpca.fit(id_feats.cpu())
    ash_bound = np.percentile(skpca.components_.T, args.ash_percentile)
    lower_bound = np.percentile(skpca.components_.T, args.lower_percentile)
    upper_bound = np.percentile(skpca.components_.T, args.upper_percentile)
elif args.method == "ica":
    ica = FastICA(n_components=args.num_component)
    ica.fit(id_feats.cpu())
elif args.method == "nmf":
    nmf = NMF(n_components=args.num_component, max_iter=20000)
    nmf.fit(id_feats.cpu())
    # ash_bound = np.percentile(nmf.components_.T, args.ash_percentile)
    # lower_bound = np.percentile(nmf.components_.T, args.lower_percentile)
    # upper_bound = np.percentile(nmf.components_.T, args.upper_percentile)
    ash_bound = args.ash_percentile
    lower_bound = args.lower_percentile
    upper_bound = args.upper_percentile

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

# torch.save(id_feats, "../data/old_id_feats_cifar100.pkl")
# torch.save(ood_feats, "../data/old_ood_feats_cifar100.pkl")
# torch.save(id_feats_test, "../data/old_id_feats_test_cifar100.pkl")
# torch.save(ood1_feats, "../data/old_ood1_feats_cifar100.pkl")
# torch.save(ood2_feats, "../data/old_ood2_feats_cifar100.pkl")
# torch.save(ood3_feats, "../data/old_ood3_feats_cifar100.pkl")
# torch.save(ood4_feats, "../data/old_ood4_feats_cifar100.pkl")
# torch.save(ood5_feats, "../data/old_ood5_feats_cifar100.pkl")
# torch.save(ood6_feats, "../data/old_ood6_feats_cifar100.pkl")

# torch.save(nmf_components, "../data/old_nmf_components_cifar100.pkl")
# torch.save(nmf_id_feats, "../data/old_nmf_id_feats_cifar100.pkl")
# torch.save(nmf_ood_feats, "../data/old_nmf_ood_feats_cifar100.pkl")
# torch.save(nmf_id_feats_test, "../data/old_nmf_id_feats_test_cifar100.pkl")
# torch.save(nmf_ood1_feats, "../data/old_nmf_ood1_feats_cifar100.pkl")
# torch.save(nmf_ood2_feats, "../data/old_nmf_ood2_feats_cifar100.pkl")
# torch.save(nmf_ood3_feats, "../data/old_nmf_ood3_feats_cifar100.pkl")
# torch.save(nmf_ood4_feats, "../data/old_nmf_ood4_feats_cifar100.pkl")
# torch.save(nmf_ood5_feats, "../data/old_nmf_ood5_feats_cifar100.pkl")
# torch.save(nmf_ood6_feats, "../data/old_nmf_ood6_feats_cifar100.pkl")

# cifar100
# torch.save(id_feats, "../data/dice_old_id_feats_cifar100.pkl")
# torch.save(ood_feats, "../data/dice_old_ood_feats_cifar100.pkl")
# torch.save(id_feats_test, "../data/dice_old_id_feats_test_cifar100.pkl")
# torch.save(ood1_feats, "../data/dice_old_ood1_feats_cifar100.pkl")
# torch.save(ood2_feats, "../data/dice_old_ood2_feats_cifar100.pkl")
# torch.save(ood3_feats, "../data/dice_old_ood3_feats_cifar100.pkl")
# torch.save(ood4_feats, "../data/dice_old_ood4_feats_cifar100.pkl")
# torch.save(ood5_feats, "../data/dice_old_ood5_feats_cifar100.pkl")
# torch.save(ood6_feats, "../data/dice_old_ood6_feats_cifar100.pkl")

# torch.save(nmf_components, "../data/dice_old_nmf_components_cifar100.pkl")
# torch.save(nmf_id_feats, "../data/dice_old_nmf_id_feats_cifar100.pkl")
# torch.save(nmf_ood_feats, "../data/dice_old_nmf_ood_feats_cifar100.pkl")
# torch.save(nmf_id_feats_test, "../data/dice_old_nmf_id_feats_test_cifar100.pkl")
# torch.save(nmf_ood1_feats, "../data/dice_old_nmf_ood1_feats_cifar100.pkl")
# torch.save(nmf_ood2_feats, "../data/dice_old_nmf_ood2_feats_cifar100.pkl")
# torch.save(nmf_ood3_feats, "../data/dice_old_nmf_ood3_feats_cifar100.pkl")
# torch.save(nmf_ood4_feats, "../data/dice_old_nmf_ood4_feats_cifar100.pkl")
# torch.save(nmf_ood5_feats, "../data/dice_old_nmf_ood5_feats_cifar100.pkl")
# torch.save(nmf_ood6_feats, "../data/dice_old_nmf_ood6_feats_cifar100.pkl")
# cifar100

# cifar10
# torch.save(id_feats, "../data/dice_old_id_feats_cifar10.pkl")
# torch.save(ood_feats, "../data/dice_old_ood_feats_cifar10.pkl")
# torch.save(id_feats_test, "../data/dice_old_id_feats_test_cifar10.pkl")
# torch.save(ood1_feats, "../data/dice_old_ood1_feats_cifar10.pkl")
# torch.save(ood2_feats, "../data/dice_old_ood2_feats_cifar10.pkl")
# torch.save(ood3_feats, "../data/dice_old_ood3_feats_cifar10.pkl")
# torch.save(ood4_feats, "../data/dice_old_ood4_feats_cifar10.pkl")
# torch.save(ood5_feats, "../data/dice_old_ood5_feats_cifar10.pkl")
# torch.save(ood6_feats, "../data/dice_old_ood6_feats_cifar10.pkl")

# torch.save(nmf_components, "../data/dice_old_nmf_components_cifar10.pkl")
# torch.save(nmf_id_feats, "../data/dice_old_nmf_id_feats_cifar10.pkl")
# torch.save(nmf_ood_feats, "../data/dice_old_nmf_ood_feats_cifar10.pkl")
# torch.save(nmf_id_feats_test, "../data/dice_old_nmf_id_feats_test_cifar10.pkl")
# torch.save(nmf_ood1_feats, "../data/dice_old_nmf_ood1_feats_cifar10.pkl")
# torch.save(nmf_ood2_feats, "../data/dice_old_nmf_ood2_feats_cifar10.pkl")
# torch.save(nmf_ood3_feats, "../data/dice_old_nmf_ood3_feats_cifar10.pkl")
# torch.save(nmf_ood4_feats, "../data/dice_old_nmf_ood4_feats_cifar10.pkl")
# torch.save(nmf_ood5_feats, "../data/dice_old_nmf_ood5_feats_cifar10.pkl")
# torch.save(nmf_ood6_feats, "../data/dice_old_nmf_ood6_feats_cifar10.pkl")
# cifar10

# def eval_datasets(clamp0_1, clamp1_1, clamp2_1, clamp3_1, clamp4_1, clamp5_1, clamp6_1, clamp7_1, clamp8_1, clamp9_1, clamp10_1, clamp11_1, clamp12_1, clamp13_1, clamp14_1, clamp0_2, clamp1_2, clamp2_2, clamp3_2, clamp4_2, clamp5_2, clamp6_2, clamp7_2, clamp8_2, clamp9_2, clamp10_2, clamp11_2, clamp12_2, clamp13_2, clamp14_2, scale):
def eval_datasets(clamp0_1, clamp1_1, clamp2_1, clamp3_1, clamp4_1, clamp5_1, clamp6_1, clamp7_1, clamp8_1, clamp9_1, clamp10_1, clamp11_1, clamp12_1, clamp13_1, clamp14_1, clamp15_1, clamp16_1, clamp17_1, clamp18_1, clamp19_1, clamp0_2, clamp1_2, clamp2_2, clamp3_2, clamp4_2, clamp5_2, clamp6_2, clamp7_2, clamp8_2, clamp9_2, clamp10_2, clamp11_2, clamp12_2, clamp13_2, clamp14_2, clamp15_2, clamp16_2, clamp17_2, clamp18_2, clamp19_2, scale):
# def eval_datasets(clamp0_1, clamp1_1, clamp2_1, clamp3_1, clamp4_1, clamp5_1, clamp6_1, clamp7_1, clamp8_1, clamp9_1, clamp10_1, clamp11_1, clamp12_1, clamp13_1, clamp14_1, clamp15_1, clamp16_1, clamp17_1, clamp18_1, clamp19_1, clamp20_1, clamp21_1, clamp22_1, clamp23_1, clamp24_1, clamp25_1, clamp26_1, clamp27_1, clamp28_1, clamp29_1, clamp30_1, clamp31_1, clamp32_1, clamp33_1, clamp34_1, clamp0_2, clamp1_2, clamp2_2, clamp3_2, clamp4_2, clamp5_2, clamp6_2, clamp7_2, clamp8_2, clamp9_2, clamp10_2, clamp11_2, clamp12_2, clamp13_2, clamp14_2, clamp15_2, clamp16_2, clamp17_2, clamp18_2, clamp19_2, clamp20_2, clamp21_2, clamp22_2, clamp23_2, clamp24_2, clamp25_2, clamp26_2, clamp27_2, clamp28_2, clamp29_2, clamp30_2, clamp31_2, clamp32_2, clamp33_2, clamp34_2, scale):
# def eval_datasets(clamp0_1, clamp1_1, clamp2_1, clamp3_1, clamp4_1, clamp5_1, clamp6_1, clamp7_1, clamp8_1, clamp9_1, clamp10_1, clamp11_1, clamp12_1, clamp13_1, clamp14_1, clamp15_1, clamp16_1, clamp17_1, clamp18_1, clamp19_1, clamp20_1, clamp21_1, clamp22_1, clamp23_1, clamp24_1, clamp25_1, clamp26_1, clamp27_1, clamp28_1, clamp29_1, clamp30_1, clamp31_1, clamp32_1, clamp33_1, clamp34_1, clamp35_1, clamp36_1, clamp37_1, clamp38_1, clamp39_1, clamp40_1, clamp41_1, clamp42_1, clamp43_1, clamp44_1, clamp45_1, clamp46_1, clamp47_1, clamp48_1, clamp49_1, clamp0_2, clamp1_2, clamp2_2, clamp3_2, clamp4_2, clamp5_2, clamp6_2, clamp7_2, clamp8_2, clamp9_2, clamp10_2, clamp11_2, clamp12_2, clamp13_2, clamp14_2, clamp15_2, clamp16_2, clamp17_2, clamp18_2, clamp19_2, clamp20_2, clamp21_2, clamp22_2, clamp23_2, clamp24_2, clamp25_2, clamp26_2, clamp27_2, clamp28_2, clamp29_2, clamp30_2, clamp31_2, clamp32_2, clamp33_2, clamp34_2, clamp35_2, clamp36_2, clamp37_2, clamp38_2, clamp39_2, clamp40_2, clamp41_2, clamp42_2, clamp43_2, clamp44_2, clamp45_2, clamp46_2, clamp47_2, clamp48_2, clamp49_2, scale):
    global id_feats, id_feats_test, ood_feats, ood1_feats, ood2_feats, ood3_feats, ood4_feats, ood5_feats, ood6_feats
    global final_fpr, final_fpr1, final_fpr2, final_fpr3, final_fpr4, final_fpr5, final_fpr6, final_avg_fpr
    global final_auroc, final_auroc1, final_auroc2, final_auroc3, final_auroc4, final_auroc5, final_auroc6, final_avg_auroc
    m_id_feats = torch.clone(id_feats)
    m_id_feats_test = torch.clone(id_feats_test)
    m_ood_feats = torch.clone(ood_feats)
    if args.feat_noise == "gaussian":
         m_ood_feats = m_ood_feats + torch.abs(torch.empty_like(m_ood_feats).normal_(0, 1)) * weight_noise
    elif args.feat_noise == "uniform":
        m_ood_feats = m_ood_feats + torch.empty_like(m_ood_feats).uniform_(0, 1) * weight_noise
    m_ood1_feats = torch.clone(ood1_feats)
    m_ood2_feats = torch.clone(ood2_feats)
    m_ood3_feats = torch.clone(ood3_feats)
    m_ood4_feats = torch.clone(ood4_feats)
    m_ood5_feats = torch.clone(ood5_feats)
    m_ood6_feats = torch.clone(ood6_feats)

    if args.method == "nmf":
        m_id_error = m_id_feats - torch.Tensor(nmf.transform(m_id_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood_error = m_ood_feats - torch.Tensor(nmf.transform(m_ood_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_trans = torch.clone(torch.Tensor(nmf.components_.T))

    if args.react_percent == "yes":
        m_trans[:, 0] = react_percent(m_trans[:, 0], clamp0_1 * 100)
        if args.num_component >= 3:
            m_trans[:, 1] = react_percent(m_trans[:, 1], clamp1_1 * 100)
            m_trans[:, 2] = react_percent(m_trans[:, 2], clamp2_1 * 100)
        if args.num_component >= 5:
            m_trans[:, 3] = react_percent(m_trans[:, 3], clamp3_1 * 100)
            m_trans[:, 4] = react_percent(m_trans[:, 4], clamp4_1 * 100)
        if args.num_component >= 10:
            m_trans[:, 5] = react_percent(m_trans[:, 5], clamp5_1 * 100)
            m_trans[:, 6] = react_percent(m_trans[:, 6], clamp6_1 * 100)
            m_trans[:, 7] = react_percent(m_trans[:, 7], clamp7_1 * 100)
            m_trans[:, 8] = react_percent(m_trans[:, 8], clamp8_1 * 100)
            m_trans[:, 9] = react_percent(m_trans[:, 9], clamp9_1 * 100)
    elif args.react_percent == "no":
        m_trans[:, 0] = react(m_trans[:, 0], clamp0_1)
        if args.num_component >= 3:
            m_trans[:, 1] = react(m_trans[:, 1], clamp1_1)
            m_trans[:, 2] = react(m_trans[:, 2], clamp2_1)
        if args.num_component >= 5:
            m_trans[:, 3] = react(m_trans[:, 3], clamp3_1)
            m_trans[:, 4] = react(m_trans[:, 4], clamp4_1)
        if args.num_component >= 10:
            m_trans[:, 5] = react(m_trans[:, 5], clamp5_1)
            m_trans[:, 6] = react(m_trans[:, 6], clamp6_1)
            m_trans[:, 7] = react(m_trans[:, 7], clamp7_1)
            m_trans[:, 8] = react(m_trans[:, 8], clamp8_1)
            m_trans[:, 9] = react(m_trans[:, 9], clamp9_1)
        if args.num_component >= 12:
            m_trans[:, 10] = react(m_trans[:, 10], clamp10_1)
            m_trans[:, 11] = react(m_trans[:, 11], clamp11_1)
        if args.num_component >= 15:
            m_trans[:, 12] = react(m_trans[:, 12], clamp12_1)
            m_trans[:, 13] = react(m_trans[:, 13], clamp13_1)
            m_trans[:, 14] = react(m_trans[:, 14], clamp14_1)
        if args.num_component >= 20:
            m_trans[:, 15] = react(m_trans[:, 15], clamp15_1)
            m_trans[:, 16] = react(m_trans[:, 16], clamp16_1)
            m_trans[:, 17] = react(m_trans[:, 17], clamp17_1)
            m_trans[:, 18] = react(m_trans[:, 18], clamp18_1)
            m_trans[:, 19] = react(m_trans[:, 19], clamp19_1)
        if args.num_component >= 25:
            m_trans[:, 20] = react(m_trans[:, 20], clamp20_1)
            m_trans[:, 21] = react(m_trans[:, 21], clamp21_1)
            m_trans[:, 22] = react(m_trans[:, 22], clamp22_1)
            m_trans[:, 23] = react(m_trans[:, 23], clamp23_1)
            m_trans[:, 24] = react(m_trans[:, 24], clamp24_1)
        if args.num_component >= 35:
            m_trans[:, 25] = react(m_trans[:, 25], clamp25_1)
            m_trans[:, 26] = react(m_trans[:, 26], clamp26_1)
            m_trans[:, 27] = react(m_trans[:, 27], clamp27_1)
            m_trans[:, 28] = react(m_trans[:, 28], clamp28_1)
            m_trans[:, 29] = react(m_trans[:, 29], clamp29_1)
            m_trans[:, 30] = react(m_trans[:, 30], clamp30_1)
            m_trans[:, 31] = react(m_trans[:, 31], clamp31_1)
            m_trans[:, 32] = react(m_trans[:, 32], clamp32_1)
            m_trans[:, 33] = react(m_trans[:, 33], clamp33_1)
            m_trans[:, 34] = react(m_trans[:, 34], clamp34_1)
        if args.num_component >= 40:
            m_trans[:, 35] = react(m_trans[:, 35], clamp35_1)
            m_trans[:, 36] = react(m_trans[:, 36], clamp36_1)
            m_trans[:, 37] = react(m_trans[:, 37], clamp37_1)
            m_trans[:, 38] = react(m_trans[:, 38], clamp38_1)
            m_trans[:, 39] = react(m_trans[:, 39], clamp39_1)
        if args.num_component >= 45:
            m_trans[:, 40] = react(m_trans[:, 40], clamp40_1)
            m_trans[:, 41] = react(m_trans[:, 41], clamp41_1)
            m_trans[:, 42] = react(m_trans[:, 42], clamp42_1)
            m_trans[:, 43] = react(m_trans[:, 43], clamp43_1)
            m_trans[:, 44] = react(m_trans[:, 44], clamp44_1)
        if args.num_component >= 50:
            m_trans[:, 45] = react(m_trans[:, 45], clamp45_1)
            m_trans[:, 46] = react(m_trans[:, 46], clamp46_1)
            m_trans[:, 47] = react(m_trans[:, 47], clamp47_1)
            m_trans[:, 48] = react(m_trans[:, 48], clamp48_1)
            m_trans[:, 49] = react(m_trans[:, 49], clamp49_1)
        if args.num_component >= 55:
            m_trans[:, 50] = react(m_trans[:, 50], clamp50_1)
            m_trans[:, 51] = react(m_trans[:, 51], clamp51_1)
            m_trans[:, 52] = react(m_trans[:, 52], clamp52_1)
            m_trans[:, 53] = react(m_trans[:, 53], clamp53_1)
            m_trans[:, 54] = react(m_trans[:, 54], clamp54_1)
        if args.num_component >= 60:
            m_trans[:, 55] = react(m_trans[:, 55], clamp55_1)
            m_trans[:, 56] = react(m_trans[:, 56], clamp56_1)
            m_trans[:, 57] = react(m_trans[:, 57], clamp57_1)
            m_trans[:, 58] = react(m_trans[:, 58], clamp58_1)
            m_trans[:, 59] = react(m_trans[:, 59], clamp59_1)
        if args.num_component >= 65:
            m_trans[:, 60] = react(m_trans[:, 60], clamp60_1)
            m_trans[:, 61] = react(m_trans[:, 61], clamp61_1)
            m_trans[:, 62] = react(m_trans[:, 62], clamp62_1)
            m_trans[:, 63] = react(m_trans[:, 63], clamp63_1)
            m_trans[:, 64] = react(m_trans[:, 64], clamp64_1)
        if args.num_component >= 70:
            m_trans[:, 65] = react(m_trans[:, 65], clamp65_1)
            m_trans[:, 66] = react(m_trans[:, 66], clamp66_1)
            m_trans[:, 67] = react(m_trans[:, 67], clamp67_1)
            m_trans[:, 68] = react(m_trans[:, 68], clamp68_1)
            m_trans[:, 69] = react(m_trans[:, 69], clamp69_1)
        if args.num_component >= 75:
            m_trans[:, 70] = react(m_trans[:, 70], clamp70_1)
            m_trans[:, 71] = react(m_trans[:, 71], clamp71_1)
            m_trans[:, 72] = react(m_trans[:, 72], clamp72_1)
            m_trans[:, 73] = react(m_trans[:, 73], clamp73_1)
            m_trans[:, 74] = react(m_trans[:, 74], clamp74_1)
        if args.num_component >= 90:
            m_trans[:, 75] = react(m_trans[:, 75], clamp75_1)
            m_trans[:, 76] = react(m_trans[:, 76], clamp76_1)
            m_trans[:, 77] = react(m_trans[:, 77], clamp77_1)
            m_trans[:, 78] = react(m_trans[:, 78], clamp78_1)
            m_trans[:, 79] = react(m_trans[:, 79], clamp79_1)
            m_trans[:, 80] = react(m_trans[:, 80], clamp80_1)
            m_trans[:, 81] = react(m_trans[:, 81], clamp81_1)
            m_trans[:, 82] = react(m_trans[:, 82], clamp82_1)
            m_trans[:, 83] = react(m_trans[:, 83], clamp83_1)
            m_trans[:, 84] = react(m_trans[:, 84], clamp84_1)
            m_trans[:, 85] = react(m_trans[:, 85], clamp85_1)
            m_trans[:, 86] = react(m_trans[:, 86], clamp86_1)
            m_trans[:, 87] = react(m_trans[:, 87], clamp87_1)
            m_trans[:, 88] = react(m_trans[:, 88], clamp88_1)
            m_trans[:, 89] = react(m_trans[:, 89], clamp89_1)
        if args.num_component >= 110:
            m_trans[:, 90] = react(m_trans[:, 90], clamp90_1)
            m_trans[:, 91] = react(m_trans[:, 91], clamp91_1)
            m_trans[:, 92] = react(m_trans[:, 92], clamp92_1)
            m_trans[:, 93] = react(m_trans[:, 93], clamp93_1)
            m_trans[:, 94] = react(m_trans[:, 94], clamp94_1)
            m_trans[:, 95] = react(m_trans[:, 95], clamp95_1)
            m_trans[:, 96] = react(m_trans[:, 96], clamp96_1)
            m_trans[:, 97] = react(m_trans[:, 97], clamp97_1)
            m_trans[:, 98] = react(m_trans[:, 98], clamp98_1)
            m_trans[:, 99] = react(m_trans[:, 99], clamp99_1)
            m_trans[:, 100] = react(m_trans[:, 100], clamp100_1)
            m_trans[:, 101] = react(m_trans[:, 101], clamp101_1)
            m_trans[:, 102] = react(m_trans[:, 102], clamp102_1)
            m_trans[:, 103] = react(m_trans[:, 103], clamp103_1)
            m_trans[:, 104] = react(m_trans[:, 104], clamp104_1)
            m_trans[:, 105] = react(m_trans[:, 105], clamp105_1)
            m_trans[:, 106] = react(m_trans[:, 106], clamp106_1)
            m_trans[:, 107] = react(m_trans[:, 107], clamp107_1)
            m_trans[:, 108] = react(m_trans[:, 108], clamp108_1)
            m_trans[:, 109] = react(m_trans[:, 109], clamp109_1)
        if args.num_component >= 130:
            m_trans[:, 110] = react(m_trans[:, 110], clamp110_1)
            m_trans[:, 111] = react(m_trans[:, 111], clamp111_1)
            m_trans[:, 112] = react(m_trans[:, 112], clamp112_1)
            m_trans[:, 113] = react(m_trans[:, 113], clamp113_1)
            m_trans[:, 114] = react(m_trans[:, 114], clamp114_1)
            m_trans[:, 115] = react(m_trans[:, 115], clamp115_1)
            m_trans[:, 116] = react(m_trans[:, 116], clamp116_1)
            m_trans[:, 117] = react(m_trans[:, 117], clamp117_1)
            m_trans[:, 118] = react(m_trans[:, 118], clamp118_1)
            m_trans[:, 119] = react(m_trans[:, 119], clamp119_1)
            m_trans[:, 120] = react(m_trans[:, 120], clamp120_1)
            m_trans[:, 121] = react(m_trans[:, 121], clamp121_1)
            m_trans[:, 122] = react(m_trans[:, 122], clamp122_1)
            m_trans[:, 123] = react(m_trans[:, 123], clamp123_1)
            m_trans[:, 124] = react(m_trans[:, 124], clamp124_1)
            m_trans[:, 125] = react(m_trans[:, 125], clamp125_1)
            m_trans[:, 126] = react(m_trans[:, 126], clamp126_1)
            m_trans[:, 127] = react(m_trans[:, 127], clamp127_1)
            m_trans[:, 128] = react(m_trans[:, 128], clamp128_1)
            m_trans[:, 129] = react(m_trans[:, 129], clamp129_1)
        if args.num_component >= 150:
            m_trans[:, 130] = react(m_trans[:, 130], clamp130_1)
            m_trans[:, 131] = react(m_trans[:, 131], clamp131_1)
            m_trans[:, 132] = react(m_trans[:, 132], clamp132_1)
            m_trans[:, 133] = react(m_trans[:, 133], clamp133_1)
            m_trans[:, 134] = react(m_trans[:, 134], clamp134_1)
            m_trans[:, 135] = react(m_trans[:, 135], clamp135_1)
            m_trans[:, 136] = react(m_trans[:, 136], clamp136_1)
            m_trans[:, 137] = react(m_trans[:, 137], clamp137_1)
            m_trans[:, 138] = react(m_trans[:, 138], clamp138_1)
            m_trans[:, 139] = react(m_trans[:, 139], clamp139_1)
            m_trans[:, 140] = react(m_trans[:, 140], clamp140_1)
            m_trans[:, 141] = react(m_trans[:, 141], clamp141_1)
            m_trans[:, 142] = react(m_trans[:, 142], clamp142_1)
            m_trans[:, 143] = react(m_trans[:, 143], clamp143_1)
            m_trans[:, 144] = react(m_trans[:, 144], clamp144_1)
            m_trans[:, 145] = react(m_trans[:, 145], clamp145_1)
            m_trans[:, 146] = react(m_trans[:, 146], clamp146_1)
            m_trans[:, 147] = react(m_trans[:, 147], clamp147_1)
            m_trans[:, 148] = react(m_trans[:, 148], clamp148_1)
            m_trans[:, 149] = react(m_trans[:, 149], clamp149_1)
    if args.use_ash == "yes":
        if args.ash_scale == "no":
            m_trans[:, 0] = ash_s_thre(m_trans[:, 0], clamp0_2)
        elif args.ash_scale == "yes":
            m_trans[:, 0] = ash_s_thre(m_trans[:, 0], clamp0_2, clamp0_3)
        if args.num_component >= 3:
            if args.ash_scale == "no":
                m_trans[:, 1] = ash_s_thre(m_trans[:, 1], clamp1_2)
                m_trans[:, 2] = ash_s_thre(m_trans[:, 2], clamp2_2)
            elif args.ash_scale == "yes":
                m_trans[:, 1] = ash_s_thre(m_trans[:, 1], clamp1_2, clamp1_3)
                m_trans[:, 2] = ash_s_thre(m_trans[:, 2], clamp2_2, clamp2_3)
        if args.num_component >= 5:
            if args.ash_scale == "no":
                m_trans[:, 3] = ash_s_thre(m_trans[:, 3], clamp3_2)
                m_trans[:, 4] = ash_s_thre(m_trans[:, 4], clamp4_2)
            elif args.ash_scale == "yes":
                m_trans[:, 3] = ash_s_thre(m_trans[:, 3], clamp3_2, clamp3_3)
                m_trans[:, 4] = ash_s_thre(m_trans[:, 4], clamp4_2, clamp4_3)
        if args.num_component >= 10:
            if args.ash_scale == "no":
                m_trans[:, 5] = ash_s_thre(m_trans[:, 5], clamp5_2)
                m_trans[:, 6] = ash_s_thre(m_trans[:, 6], clamp6_2)
                m_trans[:, 7] = ash_s_thre(m_trans[:, 7], clamp7_2)
                m_trans[:, 8] = ash_s_thre(m_trans[:, 8], clamp8_2)
                m_trans[:, 9] = ash_s_thre(m_trans[:, 9], clamp9_2)
            elif args.ash_scale == "yes":
                m_trans[:, 5] = ash_s_thre(m_trans[:, 5], clamp5_2, clamp5_3)
                m_trans[:, 6] = ash_s_thre(m_trans[:, 6], clamp6_2, clamp6_3)
                m_trans[:, 7] = ash_s_thre(m_trans[:, 7], clamp7_2, clamp7_3)
                m_trans[:, 8] = ash_s_thre(m_trans[:, 8], clamp8_2, clamp8_3)
                m_trans[:, 9] = ash_s_thre(m_trans[:, 9], clamp9_2, clamp9_3)
        if args.num_component >= 12:
            if args.ash_scale == "no":
                m_trans[:, 10] = ash_s_thre(m_trans[:, 10], clamp10_2)
                m_trans[:, 11] = ash_s_thre(m_trans[:, 11], clamp11_2)
            elif args.ash_scale == "yes":
                m_trans[:, 10] = ash_s_thre(m_trans[:, 10], clamp10_2, clamp10_3)
                m_trans[:, 11] = ash_s_thre(m_trans[:, 11], clamp11_2, clamp11_3)
        if args.num_component >= 15:
            if args.ash_scale == "no":
                m_trans[:, 12] = ash_s_thre(m_trans[:, 12], clamp12_2)
                m_trans[:, 13] = ash_s_thre(m_trans[:, 13], clamp13_2)
                m_trans[:, 14] = ash_s_thre(m_trans[:, 14], clamp14_2)
            elif args.ash_scale == "yes":
                m_trans[:, 12] = ash_s_thre(m_trans[:, 12], clamp12_2, clamp12_3)
                m_trans[:, 13] = ash_s_thre(m_trans[:, 13], clamp13_2, clamp13_3)
                m_trans[:, 14] = ash_s_thre(m_trans[:, 14], clamp14_2, clamp14_3)
        if args.num_component >= 20:
            if args.ash_scale == "no":
                m_trans[:, 15] = ash_s_thre(m_trans[:, 15], clamp15_2)
                m_trans[:, 16] = ash_s_thre(m_trans[:, 16], clamp16_2)
                m_trans[:, 17] = ash_s_thre(m_trans[:, 17], clamp17_2)
                m_trans[:, 18] = ash_s_thre(m_trans[:, 18], clamp18_2)
                m_trans[:, 19] = ash_s_thre(m_trans[:, 19], clamp19_2)
            elif args.ash_scale == "yes":
                m_trans[:, 15] = ash_s_thre(m_trans[:, 15], clamp15_2, clamp15_3)
                m_trans[:, 16] = ash_s_thre(m_trans[:, 16], clamp16_2, clamp16_3)
                m_trans[:, 17] = ash_s_thre(m_trans[:, 17], clamp17_2, clamp17_3)
                m_trans[:, 18] = ash_s_thre(m_trans[:, 18], clamp18_2, clamp18_3)
                m_trans[:, 19] = ash_s_thre(m_trans[:, 19], clamp19_2, clamp19_3)
        if args.num_component >= 25:
            if args.ash_scale == "no":
                m_trans[:, 20] = ash_s_thre(m_trans[:, 20], clamp20_2)
                m_trans[:, 21] = ash_s_thre(m_trans[:, 21], clamp21_2)
                m_trans[:, 22] = ash_s_thre(m_trans[:, 22], clamp22_2)
                m_trans[:, 23] = ash_s_thre(m_trans[:, 23], clamp23_2)
                m_trans[:, 24] = ash_s_thre(m_trans[:, 24], clamp24_2)
            elif args.ash_scale == "yes":
                m_trans[:, 20] = ash_s_thre(m_trans[:, 20], clamp20_2, clamp20_3)
                m_trans[:, 21] = ash_s_thre(m_trans[:, 21], clamp21_2, clamp21_3)
                m_trans[:, 22] = ash_s_thre(m_trans[:, 22], clamp22_2, clamp22_3)
                m_trans[:, 23] = ash_s_thre(m_trans[:, 23], clamp23_2, clamp23_3)
                m_trans[:, 24] = ash_s_thre(m_trans[:, 24], clamp24_2, clamp24_3)
        if args.num_component >= 35:
            if args.ash_scale == "no":
                m_trans[:, 25] = ash_s_thre(m_trans[:, 25], clamp25_2)
                m_trans[:, 26] = ash_s_thre(m_trans[:, 26], clamp26_2)
                m_trans[:, 27] = ash_s_thre(m_trans[:, 27], clamp27_2)
                m_trans[:, 28] = ash_s_thre(m_trans[:, 28], clamp28_2)
                m_trans[:, 29] = ash_s_thre(m_trans[:, 29], clamp29_2)
                m_trans[:, 30] = ash_s_thre(m_trans[:, 30], clamp30_2)
                m_trans[:, 31] = ash_s_thre(m_trans[:, 31], clamp31_2)
                m_trans[:, 32] = ash_s_thre(m_trans[:, 32], clamp32_2)
                m_trans[:, 33] = ash_s_thre(m_trans[:, 33], clamp33_2)
                m_trans[:, 34] = ash_s_thre(m_trans[:, 34], clamp34_2)
            elif args.ash_scale == "yes":
                m_trans[:, 25] = ash_s_thre(m_trans[:, 25], clamp25_2, clamp25_3)
                m_trans[:, 26] = ash_s_thre(m_trans[:, 26], clamp26_2, clamp26_3)
                m_trans[:, 27] = ash_s_thre(m_trans[:, 27], clamp27_2, clamp27_3)
                m_trans[:, 28] = ash_s_thre(m_trans[:, 28], clamp28_2, clamp28_3)
                m_trans[:, 29] = ash_s_thre(m_trans[:, 29], clamp29_2, clamp29_3)
                m_trans[:, 30] = ash_s_thre(m_trans[:, 30], clamp30_2, clamp30_3)
                m_trans[:, 31] = ash_s_thre(m_trans[:, 31], clamp31_2, clamp31_3)
                m_trans[:, 32] = ash_s_thre(m_trans[:, 32], clamp32_2, clamp32_3)
                m_trans[:, 33] = ash_s_thre(m_trans[:, 33], clamp33_2, clamp33_3)
                m_trans[:, 34] = ash_s_thre(m_trans[:, 34], clamp34_2, clamp34_3)
        if args.num_component >= 40:
            m_trans[:, 35] = ash_s_thre(m_trans[:, 35], clamp35_2)
            m_trans[:, 36] = ash_s_thre(m_trans[:, 36], clamp36_2)
            m_trans[:, 37] = ash_s_thre(m_trans[:, 37], clamp37_2)
            m_trans[:, 38] = ash_s_thre(m_trans[:, 38], clamp38_2)
            m_trans[:, 39] = ash_s_thre(m_trans[:, 39], clamp39_2)
        if args.num_component >= 45:
            m_trans[:, 40] = ash_s_thre(m_trans[:, 40], clamp40_2)
            m_trans[:, 41] = ash_s_thre(m_trans[:, 41], clamp41_2)
            m_trans[:, 42] = ash_s_thre(m_trans[:, 42], clamp42_2)
            m_trans[:, 43] = ash_s_thre(m_trans[:, 43], clamp43_2)
            m_trans[:, 44] = ash_s_thre(m_trans[:, 44], clamp44_2)
        if args.num_component >= 50:
            m_trans[:, 45] = ash_s_thre(m_trans[:, 45], clamp45_2)
            m_trans[:, 46] = ash_s_thre(m_trans[:, 46], clamp46_2)
            m_trans[:, 47] = ash_s_thre(m_trans[:, 47], clamp47_2)
            m_trans[:, 48] = ash_s_thre(m_trans[:, 48], clamp48_2)
            m_trans[:, 49] = ash_s_thre(m_trans[:, 49], clamp49_2)
        if args.num_component >= 55:
            m_trans[:, 50] = ash_s_thre(m_trans[:, 50], clamp50_2)
            m_trans[:, 51] = ash_s_thre(m_trans[:, 51], clamp51_2)
            m_trans[:, 52] = ash_s_thre(m_trans[:, 52], clamp52_2)
            m_trans[:, 53] = ash_s_thre(m_trans[:, 53], clamp53_2)
            m_trans[:, 54] = ash_s_thre(m_trans[:, 54], clamp54_2)
        if args.num_component >= 60:
            m_trans[:, 55] = ash_s_thre(m_trans[:, 55], clamp55_2)
            m_trans[:, 56] = ash_s_thre(m_trans[:, 56], clamp56_2)
            m_trans[:, 57] = ash_s_thre(m_trans[:, 57], clamp57_2)
            m_trans[:, 58] = ash_s_thre(m_trans[:, 58], clamp58_2)
            m_trans[:, 59] = ash_s_thre(m_trans[:, 59], clamp59_2)
        if args.num_component >= 65:
            m_trans[:, 60] = ash_s_thre(m_trans[:, 60], clamp60_2)
            m_trans[:, 61] = ash_s_thre(m_trans[:, 61], clamp61_2)
            m_trans[:, 62] = ash_s_thre(m_trans[:, 62], clamp62_2)
            m_trans[:, 63] = ash_s_thre(m_trans[:, 63], clamp63_2)
            m_trans[:, 64] = ash_s_thre(m_trans[:, 64], clamp64_2)
        if args.num_component >= 70:
            m_trans[:, 65] = ash_s_thre(m_trans[:, 65], clamp65_2)
            m_trans[:, 66] = ash_s_thre(m_trans[:, 66], clamp66_2)
            m_trans[:, 67] = ash_s_thre(m_trans[:, 67], clamp67_2)
            m_trans[:, 68] = ash_s_thre(m_trans[:, 68], clamp68_2)
            m_trans[:, 69] = ash_s_thre(m_trans[:, 69], clamp69_2)
        if args.num_component >= 75:
            m_trans[:, 70] = ash_s_thre(m_trans[:, 70], clamp70_2)
            m_trans[:, 71] = ash_s_thre(m_trans[:, 71], clamp71_2)
            m_trans[:, 72] = ash_s_thre(m_trans[:, 72], clamp72_2)
            m_trans[:, 73] = ash_s_thre(m_trans[:, 73], clamp73_2)
            m_trans[:, 74] = ash_s_thre(m_trans[:, 74], clamp74_2)
        if args.num_component >= 90:
            m_trans[:, 75] = ash_s_thre(m_trans[:, 75], clamp75_2)
            m_trans[:, 76] = ash_s_thre(m_trans[:, 76], clamp76_2)
            m_trans[:, 77] = ash_s_thre(m_trans[:, 77], clamp77_2)
            m_trans[:, 78] = ash_s_thre(m_trans[:, 78], clamp78_2)
            m_trans[:, 79] = ash_s_thre(m_trans[:, 79], clamp79_2)
            m_trans[:, 80] = ash_s_thre(m_trans[:, 80], clamp80_2)
            m_trans[:, 81] = ash_s_thre(m_trans[:, 81], clamp81_2)
            m_trans[:, 82] = ash_s_thre(m_trans[:, 82], clamp82_2)
            m_trans[:, 83] = ash_s_thre(m_trans[:, 83], clamp83_2)
            m_trans[:, 84] = ash_s_thre(m_trans[:, 84], clamp84_2)
            m_trans[:, 85] = ash_s_thre(m_trans[:, 85], clamp85_2)
            m_trans[:, 86] = ash_s_thre(m_trans[:, 86], clamp86_2)
            m_trans[:, 87] = ash_s_thre(m_trans[:, 87], clamp87_2)
            m_trans[:, 88] = ash_s_thre(m_trans[:, 88], clamp88_2)
            m_trans[:, 89] = ash_s_thre(m_trans[:, 89], clamp89_2)
        if args.num_component >= 110:
            m_trans[:, 90] = ash_s_thre(m_trans[:, 90], clamp90_2)
            m_trans[:, 91] = ash_s_thre(m_trans[:, 91], clamp91_2)
            m_trans[:, 92] = ash_s_thre(m_trans[:, 92], clamp92_2)
            m_trans[:, 93] = ash_s_thre(m_trans[:, 93], clamp93_2)
            m_trans[:, 94] = ash_s_thre(m_trans[:, 94], clamp94_2)
            m_trans[:, 95] = ash_s_thre(m_trans[:, 95], clamp95_2)
            m_trans[:, 96] = ash_s_thre(m_trans[:, 96], clamp96_2)
            m_trans[:, 97] = ash_s_thre(m_trans[:, 97], clamp97_2)
            m_trans[:, 98] = ash_s_thre(m_trans[:, 98], clamp98_2)
            m_trans[:, 99] = ash_s_thre(m_trans[:, 99], clamp99_2)
            m_trans[:, 100] = ash_s_thre(m_trans[:, 100], clamp100_2)
            m_trans[:, 101] = ash_s_thre(m_trans[:, 101], clamp101_2)
            m_trans[:, 102] = ash_s_thre(m_trans[:, 102], clamp102_2)
            m_trans[:, 103] = ash_s_thre(m_trans[:, 103], clamp103_2)
            m_trans[:, 104] = ash_s_thre(m_trans[:, 104], clamp104_2)
            m_trans[:, 105] = ash_s_thre(m_trans[:, 105], clamp105_2)
            m_trans[:, 106] = ash_s_thre(m_trans[:, 106], clamp106_2)
            m_trans[:, 107] = ash_s_thre(m_trans[:, 107], clamp107_2)
            m_trans[:, 108] = ash_s_thre(m_trans[:, 108], clamp108_2)
            m_trans[:, 109] = ash_s_thre(m_trans[:, 109], clamp109_2)
        if args.num_component >= 130:
            m_trans[:, 110] = ash_s_thre(m_trans[:, 110], clamp110_2)
            m_trans[:, 111] = ash_s_thre(m_trans[:, 111], clamp111_2)
            m_trans[:, 112] = ash_s_thre(m_trans[:, 112], clamp112_2)
            m_trans[:, 113] = ash_s_thre(m_trans[:, 113], clamp113_2)
            m_trans[:, 114] = ash_s_thre(m_trans[:, 114], clamp114_2)
            m_trans[:, 115] = ash_s_thre(m_trans[:, 115], clamp115_2)
            m_trans[:, 116] = ash_s_thre(m_trans[:, 116], clamp116_2)
            m_trans[:, 117] = ash_s_thre(m_trans[:, 117], clamp117_2)
            m_trans[:, 118] = ash_s_thre(m_trans[:, 118], clamp118_2)
            m_trans[:, 119] = ash_s_thre(m_trans[:, 119], clamp119_2)
            m_trans[:, 120] = ash_s_thre(m_trans[:, 120], clamp120_2)
            m_trans[:, 121] = ash_s_thre(m_trans[:, 121], clamp121_2)
            m_trans[:, 122] = ash_s_thre(m_trans[:, 122], clamp122_2)
            m_trans[:, 123] = ash_s_thre(m_trans[:, 123], clamp123_2)
            m_trans[:, 124] = ash_s_thre(m_trans[:, 124], clamp124_2)
            m_trans[:, 125] = ash_s_thre(m_trans[:, 125], clamp125_2)
            m_trans[:, 126] = ash_s_thre(m_trans[:, 126], clamp126_2)
            m_trans[:, 127] = ash_s_thre(m_trans[:, 127], clamp127_2)
            m_trans[:, 128] = ash_s_thre(m_trans[:, 128], clamp128_2)
            m_trans[:, 129] = ash_s_thre(m_trans[:, 129], clamp129_2)
        if args.num_component >= 150:
            m_trans[:, 130] = ash_s_thre(m_trans[:, 130], clamp130_2)
            m_trans[:, 131] = ash_s_thre(m_trans[:, 131], clamp131_2)
            m_trans[:, 132] = ash_s_thre(m_trans[:, 132], clamp132_2)
            m_trans[:, 133] = ash_s_thre(m_trans[:, 133], clamp133_2)
            m_trans[:, 134] = ash_s_thre(m_trans[:, 134], clamp134_2)
            m_trans[:, 135] = ash_s_thre(m_trans[:, 135], clamp135_2)
            m_trans[:, 136] = ash_s_thre(m_trans[:, 136], clamp136_2)
            m_trans[:, 137] = ash_s_thre(m_trans[:, 137], clamp137_2)
            m_trans[:, 138] = ash_s_thre(m_trans[:, 138], clamp138_2)
            m_trans[:, 139] = ash_s_thre(m_trans[:, 139], clamp139_2)
            m_trans[:, 140] = ash_s_thre(m_trans[:, 140], clamp140_2)
            m_trans[:, 141] = ash_s_thre(m_trans[:, 141], clamp141_2)
            m_trans[:, 142] = ash_s_thre(m_trans[:, 142], clamp142_2)
            m_trans[:, 143] = ash_s_thre(m_trans[:, 143], clamp143_2)
            m_trans[:, 144] = ash_s_thre(m_trans[:, 144], clamp144_2)
            m_trans[:, 145] = ash_s_thre(m_trans[:, 145], clamp145_2)
            m_trans[:, 146] = ash_s_thre(m_trans[:, 146], clamp146_2)
            m_trans[:, 147] = ash_s_thre(m_trans[:, 147], clamp147_2)
            m_trans[:, 148] = ash_s_thre(m_trans[:, 148], clamp148_2)
            m_trans[:, 149] = ash_s_thre(m_trans[:, 149], clamp149_2)
    elif args.use_ash == "no":
        m_trans = m_trans * torch.exp(torch.Tensor([1]))

    if args.scale == "yes":
        m_trans = m_trans * torch.exp(torch.Tensor([scale]))

    if args.method == "nmf":
        m_id_feats = torch.Tensor(nmf.transform(m_id_feats.cpu())).mm(m_trans.T).cuda() + m_id_error
        m_ood_feats = torch.Tensor(nmf.transform(m_ood_feats.cpu())).mm(m_trans.T).cuda() + m_ood_error
        
        # m_id_feats = nmf_relu(m_id_feats)
        # m_ood_feats = nmf_relu(m_ood_feats)

    if args.model == "resnet50" or args.model == "densenet_dice" or args.model == "mobilenet":
        if args.softmax_temperature == "no":
            m_id_logits = model.fc(m_id_feats)
            m_ood_logits = model.fc(m_ood_feats)
        elif args.softmax_temperature == "yes":
            m_id_logits = model.fc(m_id_feats) * temp
            m_ood_logits = model.fc(m_ood_feats) * temp
    elif args.model == "wideresnet":
        if args.softmax_temperature == "no":
            m_id_logits = model.linear(m_id_feats)
            m_ood_logits = model.linear(m_ood_feats)
        elif args.softmax_temperature == "yes":
            m_id_logits = model.linear(m_id_feats) * temp
            m_ood_logits = model.linear(m_ood_feats) * temp

    m_id_score =  - torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
    m_ood_score =  - torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()
    if torch.isnan(torch.Tensor(m_id_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood_score)).sum() != 0:
        return -1e9
        # raise ValueError("The tensor has nan or inf elements")

    # index = torch.randperm(100000)[:50000]
    _, index = torch.sort(torch.Tensor(m_ood_score))
    index = index[:50000]
    if args.feat_margin == "one":
        margin = torch.linalg.norm(m_id_feats - m_ood_feats[index], ord=1).cuda()
        margin = (margin / (m_id_feats.norm() + m_ood_feats.norm())).item()
    elif args.feat_margin == "two":
        margin = torch.linalg.norm(m_id_feats - m_ood_feats[index], ord=2).cuda()
        margin = (margin / (m_id_feats.norm() + m_ood_feats.norm())).item()
    elif args.feat_margin == "fro":
        margin = torch.linalg.norm(m_id_feats - m_ood_feats[index], ord="fro").cuda()
        margin = (margin / (m_id_feats.norm() + m_ood_feats.norm())).item()
    elif args.feat_margin == "none":
        margin = 0

    margin = margin * args.margin_scale
    
    fpr, auroc, aupr = score_get_and_print_results(log, m_id_score, m_ood_score)
    if fpr < final_fpr:
        m_id_error_test = m_id_feats_test - torch.Tensor(nmf.transform(m_id_feats_test.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood1_error = m_ood1_feats - torch.Tensor(nmf.transform(m_ood1_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood2_error = m_ood2_feats - torch.Tensor(nmf.transform(m_ood2_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood3_error = m_ood3_feats - torch.Tensor(nmf.transform(m_ood3_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood4_error = m_ood4_feats - torch.Tensor(nmf.transform(m_ood4_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood5_error = m_ood5_feats - torch.Tensor(nmf.transform(m_ood5_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        m_ood6_error = m_ood6_feats - torch.Tensor(nmf.transform(m_ood6_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()

        m_id_feats_test = torch.Tensor(nmf.transform(m_id_feats_test.cpu())).mm(m_trans.T).cuda() + m_id_error_test
        m_ood1_feats = torch.Tensor(nmf.transform(m_ood1_feats.cpu())).mm(m_trans.T).cuda() + m_ood1_error
        m_ood2_feats = torch.Tensor(nmf.transform(m_ood2_feats.cpu())).mm(m_trans.T).cuda() + m_ood2_error
        m_ood3_feats = torch.Tensor(nmf.transform(m_ood3_feats.cpu())).mm(m_trans.T).cuda() + m_ood3_error
        m_ood4_feats = torch.Tensor(nmf.transform(m_ood4_feats.cpu())).mm(m_trans.T).cuda() + m_ood4_error
        m_ood5_feats = torch.Tensor(nmf.transform(m_ood5_feats.cpu())).mm(m_trans.T).cuda() + m_ood5_error
        m_ood6_feats = torch.Tensor(nmf.transform(m_ood6_feats.cpu())).mm(m_trans.T).cuda() + m_ood6_error
        # m_id_feats_test = nmf_relu(m_id_feats_test)
        # m_ood1_feats = nmf_relu(m_ood1_feats)
        # m_ood2_feats = nmf_relu(m_ood2_feats)
        # m_ood3_feats = nmf_relu(m_ood3_feats)
        # m_ood4_feats = nmf_relu(m_ood4_feats)
        # m_ood5_feats = nmf_relu(m_ood5_feats)
        # m_ood6_feats = nmf_relu(m_ood6_feats)

        m_id_feats_test = ash_s(m_id_feats_test, 10)
        m_ood1_feats = ash_s(m_ood1_feats, 10)
        m_ood2_feats = ash_s(m_ood2_feats, 10)
        m_ood3_feats = ash_s(m_ood3_feats, 10)
        m_ood4_feats = ash_s(m_ood4_feats, 10)
        m_ood5_feats = ash_s(m_ood5_feats, 10)
        m_ood6_feats = ash_s(m_ood6_feats, 10)

        # m_id_feats_test = ash_s(m_id_feats_test, 15)
        # m_ood1_feats = ash_s(m_ood1_feats, 15)
        # m_ood2_feats = ash_s(m_ood2_feats, 15)
        # m_ood3_feats = ash_s(m_ood3_feats, 15)
        # m_ood4_feats = ash_s(m_ood4_feats, 15)
        # m_ood5_feats = ash_s(m_ood5_feats, 15)
        # m_ood6_feats = ash_s(m_ood6_feats, 15)

        # m_id_feats_test = ash_s(m_id_feats_test, 65)
        # m_ood1_feats = ash_s(m_ood1_feats, 65)
        # m_ood2_feats = ash_s(m_ood2_feats, 65)
        # m_ood3_feats = ash_s(m_ood3_feats, 65)
        # m_ood4_feats = ash_s(m_ood4_feats, 65)
        # m_ood5_feats = ash_s(m_ood5_feats, 65)
        # m_ood6_feats = ash_s(m_ood6_feats, 65)

        # m_id_feats_test = ash_s(m_id_feats_test, 70)
        # m_ood1_feats = ash_s(m_ood1_feats, 70)
        # m_ood2_feats = ash_s(m_ood2_feats, 70)
        # m_ood3_feats = ash_s(m_ood3_feats, 70)
        # m_ood4_feats = ash_s(m_ood4_feats, 70)
        # m_ood5_feats = ash_s(m_ood5_feats, 70)
        # m_ood6_feats = ash_s(m_ood6_feats, 70)

        # m_id_feats_test = ash_s(m_id_feats_test, 80)
        # m_ood1_feats = ash_s(m_ood1_feats, 80)
        # m_ood2_feats = ash_s(m_ood2_feats, 80)
        # m_ood3_feats = ash_s(m_ood3_feats, 80)
        # m_ood4_feats = ash_s(m_ood4_feats, 80)
        # m_ood5_feats = ash_s(m_ood5_feats, 80)
        # m_ood6_feats = ash_s(m_ood6_feats, 80)


        # m_id_feats_test = ash_s(m_id_feats_test, 85)
        # m_ood1_feats = ash_s(m_ood1_feats, 85)
        # m_ood2_feats = ash_s(m_ood2_feats, 85)
        # m_ood3_feats = ash_s(m_ood3_feats, 85)
        # m_ood4_feats = ash_s(m_ood4_feats, 85)
        # m_ood5_feats = ash_s(m_ood5_feats, 85)
        # m_ood6_feats = ash_s(m_ood6_feats, 85)

        # m_id_feats_test = ash_s(m_id_feats_test, 90)
        # m_ood1_feats = ash_s(m_ood1_feats, 90)
        # m_ood2_feats = ash_s(m_ood2_feats, 90)
        # m_ood3_feats = ash_s(m_ood3_feats, 90)
        # m_ood4_feats = ash_s(m_ood4_feats, 90)
        # m_ood5_feats = ash_s(m_ood5_feats, 90)
        # m_ood6_feats = ash_s(m_ood6_feats, 90)

        # m_id_feats_test = ash_s(m_id_feats_test, 95)
        # m_ood1_feats = ash_s(m_ood1_feats, 95)
        # m_ood2_feats = ash_s(m_ood2_feats, 95)
        # m_ood3_feats = ash_s(m_ood3_feats, 95)
        # m_ood4_feats = ash_s(m_ood4_feats, 95)
        # m_ood5_feats = ash_s(m_ood5_feats, 95)
        # m_ood6_feats = ash_s(m_ood6_feats, 95)

        if args.model == "resnet50" or args.model == "densenet_dice" or args.model == "mobilenet":
            if args.softmax_temperature == "no":
                m_id_logits_test = model.fc(m_id_feats_test)
                m_ood1_logits = model.fc(m_ood1_feats)
                m_ood2_logits = model.fc(m_ood2_feats)
                m_ood3_logits = model.fc(m_ood3_feats)
                m_ood4_logits = model.fc(m_ood4_feats)
                m_ood5_logits = model.fc(m_ood5_feats)
                m_ood6_logits = model.fc(m_ood6_feats)
            elif args.softmax_temperature == "yes":
                m_id_logits_test = model.fc(m_id_feats_test) * temp
                m_ood1_logits = model.fc(m_ood1_feats) * temp
                m_ood2_logits = model.fc(m_ood2_feats) * temp
                m_ood3_logits = model.fc(m_ood3_feats) * temp
                m_ood4_logits = model.fc(m_ood4_feats) * temp
                m_ood5_logits = model.fc(m_ood5_feats) * temp
                m_ood6_logits = model.fc(m_ood6_feats) * temp
        elif args.model == "wideresnet":
            if args.softmax_temperature == "no":
                m_id_logits_test = model.linear(m_id_feats_test)
                m_ood1_logits = model.linear(m_ood1_feats)
                m_ood2_logits = model.linear(m_ood2_feats)
                m_ood3_logits = model.linear(m_ood3_feats)
                m_ood4_logits = model.linear(m_ood4_feats)
                m_ood5_logits = model.linear(m_ood5_feats)
                m_ood6_logits = model.linear(m_ood6_feats)
            elif args.softmax_temperature == "yes":
                m_id_logits_test = model.linear(m_id_feats_test) * temp
                m_ood1_logits = model.linear(m_ood1_feats) * temp
                m_ood2_logits = model.linear(m_ood2_feats) * temp
                m_ood3_logits = model.linear(m_ood3_feats) * temp
                m_ood4_logits = model.linear(m_ood4_feats) * temp
                m_ood5_logits = model.linear(m_ood5_feats) * temp
                m_ood6_logits = model.linear(m_ood6_feats) * temp

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
    if args.feat_noise == "gaussian" or args.feat_noise == "uniform":
        if args.metric == "fpr":
            return margin + weight_noise - fpr
        elif args.metric == "auroc":
            return margin + weight_noise + auroc
        elif args.metric == "both":
            return margin + weight_noise + auroc - fpr
    elif args.feat_noise == "none":
        if args.metric == "fpr":
            return margin - fpr
        elif args.metric == "auroc":
            return margin + auroc
        elif args.metric == "both":
            return margin + auroc - fpr
    # return - avg_fpr

ood_bayesian = BayesianOptimization(
    eval_datasets,
    {
    'clamp0_1': (lower_bound, upper_bound),
    'clamp1_1': (lower_bound, upper_bound),
    'clamp2_1': (lower_bound, upper_bound),
    'clamp3_1': (lower_bound, upper_bound),
    'clamp4_1': (lower_bound, upper_bound),
    'clamp5_1': (lower_bound, upper_bound),
    'clamp6_1': (lower_bound, upper_bound),
    'clamp7_1': (lower_bound, upper_bound),
    'clamp8_1': (lower_bound, upper_bound),
    'clamp9_1': (lower_bound, upper_bound),
    'clamp10_1': (lower_bound, upper_bound),
    'clamp11_1': (lower_bound, upper_bound),
    'clamp12_1': (lower_bound, upper_bound),
    'clamp13_1': (lower_bound, upper_bound),
    'clamp14_1': (lower_bound, upper_bound),
    'clamp15_1': (lower_bound, upper_bound),
    'clamp16_1': (lower_bound, upper_bound),
    'clamp17_1': (lower_bound, upper_bound),
    'clamp18_1': (lower_bound, upper_bound),
    'clamp19_1': (lower_bound, upper_bound),
    # 'clamp20_1': (lower_bound, upper_bound),
    # 'clamp21_1': (lower_bound, upper_bound),
    # 'clamp22_1': (lower_bound, upper_bound),
    # 'clamp23_1': (lower_bound, upper_bound),
    # 'clamp24_1': (lower_bound, upper_bound),
    # 'clamp25_1': (lower_bound, upper_bound),
    # 'clamp26_1': (lower_bound, upper_bound),
    # 'clamp27_1': (lower_bound, upper_bound),
    # 'clamp28_1': (lower_bound, upper_bound),
    # 'clamp29_1': (lower_bound, upper_bound),
    # 'clamp30_1': (lower_bound, upper_bound),
    # 'clamp31_1': (lower_bound, upper_bound),
    # 'clamp32_1': (lower_bound, upper_bound),
    # 'clamp33_1': (lower_bound, upper_bound),
    # 'clamp34_1': (lower_bound, upper_bound),
    # 'clamp35_1': (lower_bound, upper_bound),
    # 'clamp36_1': (lower_bound, upper_bound),
    # 'clamp37_1': (lower_bound, upper_bound),
    # 'clamp38_1': (lower_bound, upper_bound),
    # 'clamp39_1': (lower_bound, upper_bound),
    # 'clamp40_1': (lower_bound, upper_bound),
    # 'clamp41_1': (lower_bound, upper_bound),
    # 'clamp42_1': (lower_bound, upper_bound),
    # 'clamp43_1': (lower_bound, upper_bound),
    # 'clamp44_1': (lower_bound, upper_bound),
    # 'clamp45_1': (lower_bound, upper_bound),
    # 'clamp46_1': (lower_bound, upper_bound),
    # 'clamp47_1': (lower_bound, upper_bound),
    # 'clamp48_1': (lower_bound, upper_bound),
    # 'clamp49_1': (lower_bound, upper_bound),

    'scale': (0.1, 10.0),

    'clamp0_2': (0.0, ash_bound),
    'clamp1_2': (0.0, ash_bound),
    'clamp2_2': (0.0, ash_bound),
    'clamp3_2': (0.0, ash_bound),
    'clamp4_2': (0.0, ash_bound),
    'clamp5_2': (0.0, ash_bound),
    'clamp6_2': (0.0, ash_bound),
    'clamp7_2': (0.0, ash_bound),
    'clamp8_2': (0.0, ash_bound),
    'clamp9_2': (0.0, ash_bound),
    'clamp10_2': (0.0, ash_bound),
    'clamp11_2': (0.0, ash_bound),
    'clamp12_2': (0.0, ash_bound),
    'clamp13_2': (0.0, ash_bound),
    'clamp14_2': (0.0, ash_bound),
    'clamp15_2': (0.0, ash_bound),
    'clamp16_2': (0.0, ash_bound),
    'clamp17_2': (0.0, ash_bound),
    'clamp18_2': (0.0, ash_bound),
    'clamp19_2': (0.0, ash_bound),
    # 'clamp20_2': (0.0, ash_bound),
    # 'clamp21_2': (0.0, ash_bound),
    # 'clamp22_2': (0.0, ash_bound),
    # 'clamp23_2': (0.0, ash_bound),
    # 'clamp24_2': (0.0, ash_bound),
    # 'clamp25_2': (0.0, ash_bound),
    # 'clamp26_2': (0.0, ash_bound),
    # 'clamp27_2': (0.0, ash_bound),
    # 'clamp28_2': (0.0, ash_bound),
    # 'clamp29_2': (0.0, ash_bound),
    # 'clamp30_2': (0.0, ash_bound),
    # 'clamp31_2': (0.0, ash_bound),
    # 'clamp32_2': (0.0, ash_bound),
    # 'clamp33_2': (0.0, ash_bound),
    # 'clamp34_2': (0.0, ash_bound),
    # 'clamp35_2': (0.0, ash_bound),
    # 'clamp36_2': (0.0, ash_bound),
    # 'clamp37_2': (0.0, ash_bound),
    # 'clamp38_2': (0.0, ash_bound),
    # 'clamp39_2': (0.0, ash_bound),
    # 'clamp40_2': (0.0, ash_bound),
    # 'clamp41_2': (0.0, ash_bound),
    # 'clamp42_2': (0.0, ash_bound),
    # 'clamp43_2': (0.0, ash_bound),
    # 'clamp44_2': (0.0, ash_bound),
    # 'clamp45_2': (0.0, ash_bound),
    # 'clamp46_2': (0.0, ash_bound),
    # 'clamp47_2': (0.0, ash_bound),
    # 'clamp48_2': (0.0, ash_bound),
    # 'clamp49_2': (0.0, ash_bound),
    },
    allow_duplicate_points=True,
)

if args.acquisition == "ucb":
    acquisition_function = UtilityFunction(kind="ucb", kappa=args.kappa)
elif args.acquisition == "ei":
    acquisition_function = UtilityFunction(kind="ei", xi=args.xi)
elif args.acquisition == "poi":
    acquisition_function = UtilityFunction(kind="poi", xi=args.xi)

ood_bayesian.maximize(
    init_points=50,
    n_iter=5000,
    acquisition_function=acquisition_function,
)

def feats_to_score(feats, loader):
    global id_feats
    extract_feats(feats, loader)
    feats = torch.cat(feats, dim=0)
    if args.method == "pca":
        error = feats - pca.transform(feats).mm(pca.proj.T)
        trans = torch.clone(pca.proj)
    elif args.method == "skpca":
        error = feats - (torch.Tensor(skpca.transform(feats.cpu())).mm(torch.Tensor(skpca.components_)).cuda() + id_feats.mean(0))
        trans = torch.clone(torch.Tensor(skpca.components_.T))
    elif args.method == "ica":
        error = feats - (torch.Tensor(ica.transform(feats.cpu())).mm(torch.pinverse(torch.Tensor(ica.components_.T))).cuda() + id_feats.mean(0))
        trans = torch.clone(torch.Tensor(ica.components_.T))
    elif args.method == "nmf":
        error = feats - torch.Tensor(nmf.transform(feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
        trans = torch.clone(torch.Tensor(nmf.components_.T))

    if args.react_percent == "yes":
        trans[:, 0] = react_percent(trans[:, 0], ood_bayesian.max["params"]["clamp0_1"] * 100)
        if args.num_component >= 3:
            trans[:, 1] = react_percent(trans[:, 1], ood_bayesian.max["params"]["clamp1_1"] * 100)
            trans[:, 2] = react_percent(trans[:, 2], ood_bayesian.max["params"]["clamp2_1"] * 100)
        if args.num_component >= 5:
            trans[:, 3] = react_percent(trans[:, 3], ood_bayesian.max["params"]["clamp3_1"] * 100)
            trans[:, 4] = react_percent(trans[:, 4], ood_bayesian.max["params"]["clamp4_1"] * 100)
        if args.num_component >= 10:
            trans[:, 5] = react_percent(trans[:, 5], ood_bayesian.max["params"]["clamp5_1"] * 100)
            trans[:, 6] = react_percent(trans[:, 6], ood_bayesian.max["params"]["clamp6_1"] * 100)
            trans[:, 7] = react_percent(trans[:, 7], ood_bayesian.max["params"]["clamp7_1"] * 100)
            trans[:, 8] = react_percent(trans[:, 8], ood_bayesian.max["params"]["clamp8_1"] * 100)
            trans[:, 9] = react_percent(trans[:, 9], ood_bayesian.max["params"]["clamp9_1"] * 100)
        if args.num_component >= 12:
            trans[:, 10] = react_percent(trans[:, 10], ood_bayesian.max["params"]["clamp10_1"] * 100)
            trans[:, 11] = react_percent(trans[:, 11], ood_bayesian.max["params"]["clamp11_1"] * 100)
        if args.num_component >= 15:
            trans[:, 12] = react_percent(trans[:, 12], ood_bayesian.max["params"]["clamp12_1"] * 100)
            trans[:, 13] = react_percent(trans[:, 13], ood_bayesian.max["params"]["clamp13_1"] * 100)
            trans[:, 14] = react_percent(trans[:, 14], ood_bayesian.max["params"]["clamp14_1"] * 100)
    elif args.react_percent == "no":
        trans[:, 0] = react(trans[:, 0], ood_bayesian.max["params"]["clamp0_1"])
        if args.num_component >= 3:
            trans[:, 1] = react(trans[:, 1], ood_bayesian.max["params"]["clamp1_1"])
            trans[:, 2] = react(trans[:, 2], ood_bayesian.max["params"]["clamp2_1"])
        if args.num_component >= 5:
            trans[:, 3] = react(trans[:, 3], ood_bayesian.max["params"]["clamp3_1"])
            trans[:, 4] = react(trans[:, 4], ood_bayesian.max["params"]["clamp4_1"])
        if args.num_component >= 10:
            trans[:, 5] = react(trans[:, 5], ood_bayesian.max["params"]["clamp5_1"])
            trans[:, 6] = react(trans[:, 6], ood_bayesian.max["params"]["clamp6_1"])
            trans[:, 7] = react(trans[:, 7], ood_bayesian.max["params"]["clamp7_1"])
            trans[:, 8] = react(trans[:, 8], ood_bayesian.max["params"]["clamp8_1"])
            trans[:, 9] = react(trans[:, 9], ood_bayesian.max["params"]["clamp9_1"])
        if args.num_component >= 12:
            trans[:, 10] = react(trans[:, 10], ood_bayesian.max["params"]["clamp10_1"])
            trans[:, 11] = react(trans[:, 11], ood_bayesian.max["params"]["clamp11_1"])
        if args.num_component >= 15:
            trans[:, 12] = react(trans[:, 12], ood_bayesian.max["params"]["clamp12_1"])
            trans[:, 13] = react(trans[:, 13], ood_bayesian.max["params"]["clamp13_1"])
            trans[:, 14] = react(trans[:, 14], ood_bayesian.max["params"]["clamp14_1"])
        if args.num_component >= 20:
            trans[:, 15] = react(trans[:, 15], ood_bayesian.max["params"]["clamp15_1"])
            trans[:, 16] = react(trans[:, 16], ood_bayesian.max["params"]["clamp16_1"])
            trans[:, 17] = react(trans[:, 17], ood_bayesian.max["params"]["clamp17_1"])
            trans[:, 18] = react(trans[:, 18], ood_bayesian.max["params"]["clamp18_1"])
            trans[:, 19] = react(trans[:, 19], ood_bayesian.max["params"]["clamp19_1"])
        if args.num_component >= 25:
            trans[:, 20] = react(trans[:, 20], ood_bayesian.max["params"]["clamp20_1"])
            trans[:, 21] = react(trans[:, 21], ood_bayesian.max["params"]["clamp21_1"])
            trans[:, 22] = react(trans[:, 22], ood_bayesian.max["params"]["clamp22_1"])
            trans[:, 23] = react(trans[:, 23], ood_bayesian.max["params"]["clamp23_1"])
            trans[:, 24] = react(trans[:, 24], ood_bayesian.max["params"]["clamp24_1"])
        if args.num_component >= 35:
            trans[:, 25] = react(trans[:, 25], ood_bayesian.max["params"]["clamp25_1"])
            trans[:, 26] = react(trans[:, 26], ood_bayesian.max["params"]["clamp26_1"])
            trans[:, 27] = react(trans[:, 27], ood_bayesian.max["params"]["clamp27_1"])
            trans[:, 28] = react(trans[:, 28], ood_bayesian.max["params"]["clamp28_1"])
            trans[:, 29] = react(trans[:, 29], ood_bayesian.max["params"]["clamp29_1"])
            trans[:, 30] = react(trans[:, 30], ood_bayesian.max["params"]["clamp30_1"])
            trans[:, 31] = react(trans[:, 31], ood_bayesian.max["params"]["clamp31_1"])
            trans[:, 32] = react(trans[:, 32], ood_bayesian.max["params"]["clamp32_1"])
            trans[:, 33] = react(trans[:, 33], ood_bayesian.max["params"]["clamp33_1"])
            trans[:, 34] = react(trans[:, 34], ood_bayesian.max["params"]["clamp34_1"])
        if args.num_component >= 40:
            trans[:, 35] = react(trans[:, 35], ood_bayesian.max["params"]["clamp35_1"])
            trans[:, 36] = react(trans[:, 36], ood_bayesian.max["params"]["clamp36_1"])
            trans[:, 37] = react(trans[:, 37], ood_bayesian.max["params"]["clamp37_1"])
            trans[:, 38] = react(trans[:, 38], ood_bayesian.max["params"]["clamp38_1"])
            trans[:, 39] = react(trans[:, 39], ood_bayesian.max["params"]["clamp39_1"])
        if args.num_component >= 45:
            trans[:, 40] = react(trans[:, 40], ood_bayesian.max["params"]["clamp40_1"])
            trans[:, 41] = react(trans[:, 41], ood_bayesian.max["params"]["clamp41_1"])
            trans[:, 42] = react(trans[:, 42], ood_bayesian.max["params"]["clamp42_1"])
            trans[:, 43] = react(trans[:, 43], ood_bayesian.max["params"]["clamp43_1"])
            trans[:, 44] = react(trans[:, 44], ood_bayesian.max["params"]["clamp44_1"])
        if args.num_component >= 50:
            trans[:, 45] = react(trans[:, 45], ood_bayesian.max["params"]["clamp45_1"])
            trans[:, 46] = react(trans[:, 46], ood_bayesian.max["params"]["clamp46_1"])
            trans[:, 47] = react(trans[:, 47], ood_bayesian.max["params"]["clamp47_1"])
            trans[:, 48] = react(trans[:, 48], ood_bayesian.max["params"]["clamp48_1"])
            trans[:, 49] = react(trans[:, 49], ood_bayesian.max["params"]["clamp49_1"])
        if args.num_component >= 65:
            trans[:, 50] = react(trans[:, 50], ood_bayesian.max["params"]["clamp50_1"])
            trans[:, 51] = react(trans[:, 51], ood_bayesian.max["params"]["clamp51_1"])
            trans[:, 52] = react(trans[:, 52], ood_bayesian.max["params"]["clamp52_1"])
            trans[:, 53] = react(trans[:, 53], ood_bayesian.max["params"]["clamp53_1"])
            trans[:, 54] = react(trans[:, 54], ood_bayesian.max["params"]["clamp54_1"])
            trans[:, 55] = react(trans[:, 55], ood_bayesian.max["params"]["clamp55_1"])
            trans[:, 56] = react(trans[:, 56], ood_bayesian.max["params"]["clamp56_1"])
            trans[:, 57] = react(trans[:, 57], ood_bayesian.max["params"]["clamp57_1"])
            trans[:, 58] = react(trans[:, 58], ood_bayesian.max["params"]["clamp58_1"])
            trans[:, 59] = react(trans[:, 59], ood_bayesian.max["params"]["clamp59_1"])
            trans[:, 60] = react(trans[:, 60], ood_bayesian.max["params"]["clamp60_1"])
            trans[:, 61] = react(trans[:, 61], ood_bayesian.max["params"]["clamp61_1"])
            trans[:, 62] = react(trans[:, 62], ood_bayesian.max["params"]["clamp62_1"])
            trans[:, 63] = react(trans[:, 63], ood_bayesian.max["params"]["clamp63_1"])
            trans[:, 64] = react(trans[:, 64], ood_bayesian.max["params"]["clamp64_1"])
        if args.num_component >= 70:
            trans[:, 65] = react(trans[:, 65], ood_bayesian.max["params"]["clamp65_1"])
            trans[:, 66] = react(trans[:, 66], ood_bayesian.max["params"]["clamp66_1"])
            trans[:, 67] = react(trans[:, 67], ood_bayesian.max["params"]["clamp67_1"])
            trans[:, 68] = react(trans[:, 68], ood_bayesian.max["params"]["clamp68_1"])
            trans[:, 69] = react(trans[:, 69], ood_bayesian.max["params"]["clamp69_1"])
        if args.num_component >= 75:
            trans[:, 70] = react(trans[:, 70], ood_bayesian.max["params"]["clamp70_1"])
            trans[:, 71] = react(trans[:, 71], ood_bayesian.max["params"]["clamp71_1"])
            trans[:, 72] = react(trans[:, 72], ood_bayesian.max["params"]["clamp72_1"])
            trans[:, 73] = react(trans[:, 73], ood_bayesian.max["params"]["clamp73_1"])
            trans[:, 74] = react(trans[:, 74], ood_bayesian.max["params"]["clamp74_1"])
        if args.num_component >= 90:
            trans[:, 75] = react(trans[:, 75], ood_bayesian.max["params"]["clamp75_1"])
            trans[:, 76] = react(trans[:, 76], ood_bayesian.max["params"]["clamp76_1"])
            trans[:, 77] = react(trans[:, 77], ood_bayesian.max["params"]["clamp77_1"])
            trans[:, 78] = react(trans[:, 78], ood_bayesian.max["params"]["clamp78_1"])
            trans[:, 79] = react(trans[:, 79], ood_bayesian.max["params"]["clamp79_1"])
            trans[:, 80] = react(trans[:, 80], ood_bayesian.max["params"]["clamp80_1"])
            trans[:, 81] = react(trans[:, 81], ood_bayesian.max["params"]["clamp81_1"])
            trans[:, 82] = react(trans[:, 82], ood_bayesian.max["params"]["clamp82_1"])
            trans[:, 83] = react(trans[:, 83], ood_bayesian.max["params"]["clamp83_1"])
            trans[:, 84] = react(trans[:, 84], ood_bayesian.max["params"]["clamp84_1"])
            trans[:, 85] = react(trans[:, 85], ood_bayesian.max["params"]["clamp85_1"])
            trans[:, 86] = react(trans[:, 86], ood_bayesian.max["params"]["clamp86_1"])
            trans[:, 87] = react(trans[:, 87], ood_bayesian.max["params"]["clamp87_1"])
            trans[:, 88] = react(trans[:, 88], ood_bayesian.max["params"]["clamp88_1"])
            trans[:, 89] = react(trans[:, 89], ood_bayesian.max["params"]["clamp89_1"])
        if args.num_component >= 110:
            trans[:, 90] = react(trans[:, 90], ood_bayesian.max["params"]["clamp90_1"])
            trans[:, 91] = react(trans[:, 91], ood_bayesian.max["params"]["clamp91_1"])
            trans[:, 92] = react(trans[:, 92], ood_bayesian.max["params"]["clamp92_1"])
            trans[:, 93] = react(trans[:, 93], ood_bayesian.max["params"]["clamp93_1"])
            trans[:, 94] = react(trans[:, 94], ood_bayesian.max["params"]["clamp94_1"])
            trans[:, 95] = react(trans[:, 95], ood_bayesian.max["params"]["clamp95_1"])
            trans[:, 96] = react(trans[:, 96], ood_bayesian.max["params"]["clamp96_1"])
            trans[:, 97] = react(trans[:, 97], ood_bayesian.max["params"]["clamp97_1"])
            trans[:, 98] = react(trans[:, 98], ood_bayesian.max["params"]["clamp98_1"])
            trans[:, 99] = react(trans[:, 99], ood_bayesian.max["params"]["clamp99_1"])
            trans[:, 100] = react(trans[:, 100], ood_bayesian.max["params"]["clamp100_1"])
            trans[:, 101] = react(trans[:, 101], ood_bayesian.max["params"]["clamp101_1"])
            trans[:, 102] = react(trans[:, 102], ood_bayesian.max["params"]["clamp102_1"])
            trans[:, 103] = react(trans[:, 103], ood_bayesian.max["params"]["clamp103_1"])
            trans[:, 104] = react(trans[:, 104], ood_bayesian.max["params"]["clamp104_1"])
            trans[:, 105] = react(trans[:, 105], ood_bayesian.max["params"]["clamp105_1"])
            trans[:, 106] = react(trans[:, 106], ood_bayesian.max["params"]["clamp106_1"])
            trans[:, 107] = react(trans[:, 107], ood_bayesian.max["params"]["clamp107_1"])
            trans[:, 108] = react(trans[:, 108], ood_bayesian.max["params"]["clamp108_1"])
            trans[:, 109] = react(trans[:, 109], ood_bayesian.max["params"]["clamp109_1"])
        if args.num_component >= 130:
            trans[:, 110] = react(trans[:, 110], ood_bayesian.max["params"]["clamp110_1"])
            trans[:, 111] = react(trans[:, 111], ood_bayesian.max["params"]["clamp111_1"])
            trans[:, 112] = react(trans[:, 112], ood_bayesian.max["params"]["clamp112_1"])
            trans[:, 113] = react(trans[:, 113], ood_bayesian.max["params"]["clamp113_1"])
            trans[:, 114] = react(trans[:, 114], ood_bayesian.max["params"]["clamp114_1"])
            trans[:, 115] = react(trans[:, 115], ood_bayesian.max["params"]["clamp115_1"])
            trans[:, 116] = react(trans[:, 116], ood_bayesian.max["params"]["clamp116_1"])
            trans[:, 117] = react(trans[:, 117], ood_bayesian.max["params"]["clamp117_1"])
            trans[:, 118] = react(trans[:, 118], ood_bayesian.max["params"]["clamp118_1"])
            trans[:, 119] = react(trans[:, 119], ood_bayesian.max["params"]["clamp119_1"])
            trans[:, 120] = react(trans[:, 120], ood_bayesian.max["params"]["clamp120_1"])
            trans[:, 121] = react(trans[:, 121], ood_bayesian.max["params"]["clamp121_1"])
            trans[:, 122] = react(trans[:, 122], ood_bayesian.max["params"]["clamp122_1"])
            trans[:, 123] = react(trans[:, 123], ood_bayesian.max["params"]["clamp123_1"])
            trans[:, 124] = react(trans[:, 124], ood_bayesian.max["params"]["clamp124_1"])
            trans[:, 125] = react(trans[:, 125], ood_bayesian.max["params"]["clamp125_1"])
            trans[:, 126] = react(trans[:, 126], ood_bayesian.max["params"]["clamp126_1"])
            trans[:, 127] = react(trans[:, 127], ood_bayesian.max["params"]["clamp127_1"])
            trans[:, 128] = react(trans[:, 128], ood_bayesian.max["params"]["clamp128_1"])
            trans[:, 129] = react(trans[:, 129], ood_bayesian.max["params"]["clamp129_1"])
        if args.num_component >= 150:
            trans[:, 130] = react(trans[:, 130], ood_bayesian.max["params"]["clamp130_1"])
            trans[:, 131] = react(trans[:, 131], ood_bayesian.max["params"]["clamp131_1"])
            trans[:, 132] = react(trans[:, 132], ood_bayesian.max["params"]["clamp132_1"])
            trans[:, 133] = react(trans[:, 133], ood_bayesian.max["params"]["clamp133_1"])
            trans[:, 134] = react(trans[:, 134], ood_bayesian.max["params"]["clamp134_1"])
            trans[:, 135] = react(trans[:, 135], ood_bayesian.max["params"]["clamp135_1"])
            trans[:, 136] = react(trans[:, 136], ood_bayesian.max["params"]["clamp136_1"])
            trans[:, 137] = react(trans[:, 137], ood_bayesian.max["params"]["clamp137_1"])
            trans[:, 138] = react(trans[:, 138], ood_bayesian.max["params"]["clamp138_1"])
            trans[:, 139] = react(trans[:, 139], ood_bayesian.max["params"]["clamp139_1"])
            trans[:, 140] = react(trans[:, 140], ood_bayesian.max["params"]["clamp140_1"])
            trans[:, 141] = react(trans[:, 141], ood_bayesian.max["params"]["clamp141_1"])
            trans[:, 142] = react(trans[:, 142], ood_bayesian.max["params"]["clamp142_1"])
            trans[:, 143] = react(trans[:, 143], ood_bayesian.max["params"]["clamp143_1"])
            trans[:, 144] = react(trans[:, 144], ood_bayesian.max["params"]["clamp144_1"])
            trans[:, 145] = react(trans[:, 145], ood_bayesian.max["params"]["clamp145_1"])
            trans[:, 146] = react(trans[:, 146], ood_bayesian.max["params"]["clamp146_1"])
            trans[:, 147] = react(trans[:, 147], ood_bayesian.max["params"]["clamp147_1"])
            trans[:, 148] = react(trans[:, 148], ood_bayesian.max["params"]["clamp148_1"])
            trans[:, 149] = react(trans[:, 149], ood_bayesian.max["params"]["clamp149_1"])
    if args.use_ash == "yes":
        trans[:, 0] = ash_s_thre(trans[:, 0], ood_bayesian.max["params"]["clamp0_2"])
        # trans[:, 0] = ash_s_thre(trans[:, 0], ood_bayesian.max["params"]["clamp0_2"], ood_bayesian.max["params"]["clamp0_3"])
        if args.num_component >= 3:
            trans[:, 1] = ash_s_thre(trans[:, 1], ood_bayesian.max["params"]["clamp1_2"])
            trans[:, 2] = ash_s_thre(trans[:, 2], ood_bayesian.max["params"]["clamp2_2"])
            # trans[:, 1] = ash_s_thre(trans[:, 1], ood_bayesian.max["params"]["clamp1_2"], ood_bayesian.max["params"]["clamp1_3"])
            # trans[:, 2] = ash_s_thre(trans[:, 2], ood_bayesian.max["params"]["clamp2_2"], ood_bayesian.max["params"]["clamp2_3"])
        if args.num_component >= 5:
            trans[:, 3] = ash_s_thre(trans[:, 3], ood_bayesian.max["params"]["clamp3_2"])
            trans[:, 4] = ash_s_thre(trans[:, 4], ood_bayesian.max["params"]["clamp4_2"])
            # trans[:, 3] = ash_s_thre(trans[:, 3], ood_bayesian.max["params"]["clamp3_2"], ood_bayesian.max["params"]["clamp3_3"])
            # trans[:, 4] = ash_s_thre(trans[:, 4], ood_bayesian.max["params"]["clamp4_2"], ood_bayesian.max["params"]["clamp4_3"])
        if args.num_component >= 10:
            trans[:, 5] = ash_s_thre(trans[:, 5], ood_bayesian.max["params"]["clamp5_2"])
            trans[:, 6] = ash_s_thre(trans[:, 6], ood_bayesian.max["params"]["clamp6_2"])
            trans[:, 7] = ash_s_thre(trans[:, 7], ood_bayesian.max["params"]["clamp7_2"])
            trans[:, 8] = ash_s_thre(trans[:, 8], ood_bayesian.max["params"]["clamp8_2"])
            trans[:, 9] = ash_s_thre(trans[:, 9], ood_bayesian.max["params"]["clamp9_2"])
            # trans[:, 5] = ash_s_thre(trans[:, 5], ood_bayesian.max["params"]["clamp5_2"], ood_bayesian.max["params"]["clamp5_3"])
            # trans[:, 6] = ash_s_thre(trans[:, 6], ood_bayesian.max["params"]["clamp6_2"], ood_bayesian.max["params"]["clamp6_3"])
            # trans[:, 7] = ash_s_thre(trans[:, 7], ood_bayesian.max["params"]["clamp7_2"], ood_bayesian.max["params"]["clamp7_3"])
            # trans[:, 8] = ash_s_thre(trans[:, 8], ood_bayesian.max["params"]["clamp8_2"], ood_bayesian.max["params"]["clamp8_3"])
            # trans[:, 9] = ash_s_thre(trans[:, 9], ood_bayesian.max["params"]["clamp9_2"], ood_bayesian.max["params"]["clamp9_3"])
        if args.num_component >= 12:
            trans[:, 10] = ash_s_thre(trans[:, 10], ood_bayesian.max["params"]["clamp10_2"])
            trans[:, 11] = ash_s_thre(trans[:, 11], ood_bayesian.max["params"]["clamp11_2"])
        if args.num_component >= 15:
            trans[:, 12] = ash_s_thre(trans[:, 12], ood_bayesian.max["params"]["clamp12_2"])
            trans[:, 13] = ash_s_thre(trans[:, 13], ood_bayesian.max["params"]["clamp13_2"])
            trans[:, 14] = ash_s_thre(trans[:, 14], ood_bayesian.max["params"]["clamp14_2"])
        if args.num_component >= 20:
            trans[:, 15] = ash_s_thre(trans[:, 15], ood_bayesian.max["params"]["clamp15_2"])
            trans[:, 16] = ash_s_thre(trans[:, 16], ood_bayesian.max["params"]["clamp16_2"])
            trans[:, 17] = ash_s_thre(trans[:, 17], ood_bayesian.max["params"]["clamp17_2"])
            trans[:, 18] = ash_s_thre(trans[:, 18], ood_bayesian.max["params"]["clamp18_2"])
            trans[:, 19] = ash_s_thre(trans[:, 19], ood_bayesian.max["params"]["clamp19_2"])
        if args.num_component >= 25:
            trans[:, 20] = ash_s_thre(trans[:, 20], ood_bayesian.max["params"]["clamp20_2"])
            trans[:, 21] = ash_s_thre(trans[:, 21], ood_bayesian.max["params"]["clamp21_2"])
            trans[:, 22] = ash_s_thre(trans[:, 22], ood_bayesian.max["params"]["clamp22_2"])
            trans[:, 23] = ash_s_thre(trans[:, 23], ood_bayesian.max["params"]["clamp23_2"])
            trans[:, 24] = ash_s_thre(trans[:, 24], ood_bayesian.max["params"]["clamp24_2"])
        if args.num_component >= 35:
            trans[:, 25] = ash_s_thre(trans[:, 25], ood_bayesian.max["params"]["clamp25_2"])
            trans[:, 26] = ash_s_thre(trans[:, 26], ood_bayesian.max["params"]["clamp26_2"])
            trans[:, 27] = ash_s_thre(trans[:, 27], ood_bayesian.max["params"]["clamp27_2"])
            trans[:, 28] = ash_s_thre(trans[:, 28], ood_bayesian.max["params"]["clamp28_2"])
            trans[:, 29] = ash_s_thre(trans[:, 29], ood_bayesian.max["params"]["clamp29_2"])
            trans[:, 30] = ash_s_thre(trans[:, 30], ood_bayesian.max["params"]["clamp30_2"])
            trans[:, 31] = ash_s_thre(trans[:, 31], ood_bayesian.max["params"]["clamp31_2"])
            trans[:, 32] = ash_s_thre(trans[:, 32], ood_bayesian.max["params"]["clamp32_2"])
            trans[:, 33] = ash_s_thre(trans[:, 33], ood_bayesian.max["params"]["clamp33_2"])
            trans[:, 34] = ash_s_thre(trans[:, 34], ood_bayesian.max["params"]["clamp34_2"])
        if args.num_component >= 40:
            trans[:, 35] = ash_s_thre(trans[:, 35], ood_bayesian.max["params"]["clamp35_2"])
            trans[:, 36] = ash_s_thre(trans[:, 36], ood_bayesian.max["params"]["clamp36_2"])
            trans[:, 37] = ash_s_thre(trans[:, 37], ood_bayesian.max["params"]["clamp37_2"])
            trans[:, 38] = ash_s_thre(trans[:, 38], ood_bayesian.max["params"]["clamp38_2"])
            trans[:, 39] = ash_s_thre(trans[:, 39], ood_bayesian.max["params"]["clamp39_2"])
        if args.num_component >= 45:
            trans[:, 40] = ash_s_thre(trans[:, 40], ood_bayesian.max["params"]["clamp40_2"])
            trans[:, 41] = ash_s_thre(trans[:, 41], ood_bayesian.max["params"]["clamp41_2"])
            trans[:, 42] = ash_s_thre(trans[:, 42], ood_bayesian.max["params"]["clamp42_2"])
            trans[:, 43] = ash_s_thre(trans[:, 43], ood_bayesian.max["params"]["clamp43_2"])
            trans[:, 44] = ash_s_thre(trans[:, 44], ood_bayesian.max["params"]["clamp44_2"])
        if args.num_component >= 50:
            trans[:, 45] = ash_s_thre(trans[:, 45], ood_bayesian.max["params"]["clamp45_2"])
            trans[:, 46] = ash_s_thre(trans[:, 46], ood_bayesian.max["params"]["clamp46_2"])
            trans[:, 47] = ash_s_thre(trans[:, 47], ood_bayesian.max["params"]["clamp47_2"])
            trans[:, 48] = ash_s_thre(trans[:, 48], ood_bayesian.max["params"]["clamp48_2"])
            trans[:, 49] = ash_s_thre(trans[:, 49], ood_bayesian.max["params"]["clamp49_2"])
        if args.num_component >= 65:
            trans[:, 50] = ash_s_thre(trans[:, 50], ood_bayesian.max["params"]["clamp50_2"])
            trans[:, 51] = ash_s_thre(trans[:, 51], ood_bayesian.max["params"]["clamp51_2"])
            trans[:, 52] = ash_s_thre(trans[:, 52], ood_bayesian.max["params"]["clamp52_2"])
            trans[:, 53] = ash_s_thre(trans[:, 53], ood_bayesian.max["params"]["clamp53_2"])
            trans[:, 54] = ash_s_thre(trans[:, 54], ood_bayesian.max["params"]["clamp54_2"])
            trans[:, 55] = ash_s_thre(trans[:, 55], ood_bayesian.max["params"]["clamp55_2"])
            trans[:, 56] = ash_s_thre(trans[:, 56], ood_bayesian.max["params"]["clamp56_2"])
            trans[:, 57] = ash_s_thre(trans[:, 57], ood_bayesian.max["params"]["clamp57_2"])
            trans[:, 58] = ash_s_thre(trans[:, 58], ood_bayesian.max["params"]["clamp58_2"])
            trans[:, 59] = ash_s_thre(trans[:, 59], ood_bayesian.max["params"]["clamp59_2"])
            trans[:, 60] = ash_s_thre(trans[:, 60], ood_bayesian.max["params"]["clamp60_2"])
            trans[:, 61] = ash_s_thre(trans[:, 61], ood_bayesian.max["params"]["clamp61_2"])
            trans[:, 62] = ash_s_thre(trans[:, 62], ood_bayesian.max["params"]["clamp62_2"])
            trans[:, 63] = ash_s_thre(trans[:, 63], ood_bayesian.max["params"]["clamp63_2"])
            trans[:, 64] = ash_s_thre(trans[:, 64], ood_bayesian.max["params"]["clamp64_2"])
        if args.num_component >= 70:
            trans[:, 65] = ash_s_thre(trans[:, 65], ood_bayesian.max["params"]["clamp65_2"])
            trans[:, 66] = ash_s_thre(trans[:, 66], ood_bayesian.max["params"]["clamp66_2"])
            trans[:, 67] = ash_s_thre(trans[:, 67], ood_bayesian.max["params"]["clamp67_2"])
            trans[:, 68] = ash_s_thre(trans[:, 68], ood_bayesian.max["params"]["clamp68_2"])
            trans[:, 69] = ash_s_thre(trans[:, 69], ood_bayesian.max["params"]["clamp69_2"])
        if args.num_component >= 75:
            trans[:, 70] = ash_s_thre(trans[:, 70], ood_bayesian.max["params"]["clamp70_2"])
            trans[:, 71] = ash_s_thre(trans[:, 71], ood_bayesian.max["params"]["clamp71_2"])
            trans[:, 72] = ash_s_thre(trans[:, 72], ood_bayesian.max["params"]["clamp72_2"])
            trans[:, 73] = ash_s_thre(trans[:, 73], ood_bayesian.max["params"]["clamp73_2"])
            trans[:, 74] = ash_s_thre(trans[:, 74], ood_bayesian.max["params"]["clamp74_2"])

    elif args.use_ash == "no":
        trans = trans * torch.exp(torch.Tensor([1]))

    if args.scale == "yes":
        trans = trans * torch.exp(torch.Tensor([ood_bayesian.max["params"]["scale"]]))

    if args.method == "pca":
        feats = pca.transform(feats).mm(trans.T) + error
    elif args.method == "skpca":
        feats = torch.Tensor(skpca.transform(feats.cpu())).mm(trans.T).cuda() + id_feats.mean(0) + error
    elif args.method == "ica":
        feats = torch.Tensor(ica.transform(feats.cpu())).mm(torch.pinverse(trans)).cuda() + id_feats.mean(0) + error
    elif args.method == "nmf":
        feats = torch.Tensor(nmf.transform(feats.cpu())).mm(trans.T).cuda() + error

    if args.model == "densenet_dice" or args.model == "densenet_ash":
        if args.softmax_temperature == "no":
            logits = model.fc(feats)
        elif args.softmax_temperature == "yes":
            logits = model.fc(feats) * ood_bayesian.max["params"]["temp"]
    elif args.model == "densenet161":
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