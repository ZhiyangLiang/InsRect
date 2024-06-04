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
from models.densenet_dice import DenseNet3

from utils.svhn_loader import SVHN
import numpy as np
import sklearn.metrics as sk
from sklearn.cluster import k_means
from sklearn.decomposition import PCA as PCA
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import wandb
wandb.init(project="ood_detection", entity="zyliang")

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num_component", type=int)
parser.add_argument("--num_clusters", type=int)
parser.add_argument("--lower_percentile", type=float)
parser.add_argument("--upper_percentile", type=float)
parser.add_argument("--ash_percentile", type=float)
args = parser.parse_args()

final_fpr = 1.0
recall_level_default = 0.95

id_feats = []
ood_feats = []
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

log = logging.getLogger("InsRect")

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
# id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)
id_loader = torch.utils.data.DataLoader(id_data, batch_size=200, shuffle=True, num_workers=4)

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
ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=ood_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=200, shuffle=True, num_workers=4)

model = DenseNet3(100, 100).cuda()
model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

def ash_s_thre(x, threshold):
# def ash_s_thre(x, threshold, sca):
    s1 = x.sum()
    k = (x >= threshold).sum()
    t = x.view((1, -1))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum()
    scale = s1 / s2
    x *= torch.exp(scale)
    # x *= torch.exp(scale * sca)
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

def extract_feats(feats, loader, opt=0):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            if opt == 1:
                data = data + torch.empty_like(data).normal_(0, 0.005) # Gaussian Noise
            feats.append(model.get_features_fc(data))

extract_feats(id_feats, id_loader)
extract_feats(ood_feats, ood_loader)
id_feats = torch.cat(id_feats, dim=0)
ood_feats = torch.cat(ood_feats, dim=0)
centroids, labels, inertia = k_means(id_feats.cpu(), n_clusters=args.num_clusters, max_iter=20000)

id_feats_0 = id_feats[labels == 0]
id_feats_1 = id_feats[labels == 1]
id_feats_2 = id_feats[labels == 2]
id_feats_3 = id_feats[labels == 3]
id_feats_4 = id_feats[labels == 4]
id_feats_5 = id_feats[labels == 5]
id_feats_6 = id_feats[labels == 6]
id_feats_7 = id_feats[labels == 7]
id_feats_8 = id_feats[labels == 8]
id_feats_9 = id_feats[labels == 9]
if args.num_clusters > 10:
    id_feats_10 = id_feats[labels == 10]
    id_feats_11 = id_feats[labels == 11]
    id_feats_12 = id_feats[labels == 12]
    id_feats_13 = id_feats[labels == 13]
    id_feats_14 = id_feats[labels == 14]
    id_feats_15 = id_feats[labels == 15]
    id_feats_16 = id_feats[labels == 16]
    id_feats_17 = id_feats[labels == 17]
    id_feats_18 = id_feats[labels == 18]
    id_feats_19 = id_feats[labels == 19]

pca_0 = PCA(n_components=args.num_component)
pca_1 = PCA(n_components=args.num_component)
pca_2 = PCA(n_components=args.num_component)
pca_3 = PCA(n_components=args.num_component)
pca_4 = PCA(n_components=args.num_component)
pca_5 = PCA(n_components=args.num_component)
pca_6 = PCA(n_components=args.num_component)
pca_7 = PCA(n_components=args.num_component)
pca_8 = PCA(n_components=args.num_component)
pca_9 = PCA(n_components=args.num_component)
if args.num_clusters > 10:
    pca_10 = PCA(n_components=args.num_component)
    pca_11 = PCA(n_components=args.num_component)
    pca_12 = PCA(n_components=args.num_component)
    pca_13 = PCA(n_components=args.num_component)
    pca_14 = PCA(n_components=args.num_component)
    pca_15 = PCA(n_components=args.num_component)
    pca_16 = PCA(n_components=args.num_component)
    pca_17 = PCA(n_components=args.num_component)
    pca_18 = PCA(n_components=args.num_component)
    pca_19 = PCA(n_components=args.num_component)

pca_0.fit(id_feats_0.cpu())
ash_bound_0 = np.percentile(pca_0.components_.T, args.ash_percentile)
lower_bound_0 = np.percentile(pca_0.components_.T, args.lower_percentile)
upper_bound_0 = np.percentile(pca_0.components_.T, args.upper_percentile)
pca_1.fit(id_feats_1.cpu())
ash_bound_1 = np.percentile(pca_1.components_.T, args.ash_percentile)
lower_bound_1 = np.percentile(pca_1.components_.T, args.lower_percentile)
upper_bound_1 = np.percentile(pca_1.components_.T, args.upper_percentile)
pca_2.fit(id_feats_2.cpu())
ash_bound_2 = np.percentile(pca_2.components_.T, args.ash_percentile)
lower_bound_2 = np.percentile(pca_2.components_.T, args.lower_percentile)
upper_bound_2 = np.percentile(pca_2.components_.T, args.upper_percentile)
pca_3.fit(id_feats_3.cpu())
ash_bound_3 = np.percentile(pca_3.components_.T, args.ash_percentile)
lower_bound_3 = np.percentile(pca_3.components_.T, args.lower_percentile)
upper_bound_3 = np.percentile(pca_3.components_.T, args.upper_percentile)
pca_4.fit(id_feats_4.cpu())
ash_bound_4 = np.percentile(pca_4.components_.T, args.ash_percentile)
lower_bound_4 = np.percentile(pca_4.components_.T, args.lower_percentile)
upper_bound_4 = np.percentile(pca_4.components_.T, args.upper_percentile)
pca_5.fit(id_feats_5.cpu())
ash_bound_5 = np.percentile(pca_5.components_.T, args.ash_percentile)
lower_bound_5 = np.percentile(pca_5.components_.T, args.lower_percentile)
upper_bound_5 = np.percentile(pca_5.components_.T, args.upper_percentile)
pca_6.fit(id_feats_6.cpu())
ash_bound_6 = np.percentile(pca_6.components_.T, args.ash_percentile)
lower_bound_6 = np.percentile(pca_6.components_.T, args.lower_percentile)
upper_bound_6 = np.percentile(pca_6.components_.T, args.upper_percentile)
pca_7.fit(id_feats_7.cpu())
ash_bound_7 = np.percentile(pca_7.components_.T, args.ash_percentile)
lower_bound_7 = np.percentile(pca_7.components_.T, args.lower_percentile)
upper_bound_7 = np.percentile(pca_7.components_.T, args.upper_percentile)
pca_8.fit(id_feats_8.cpu())
ash_bound_8 = np.percentile(pca_8.components_.T, args.ash_percentile)
lower_bound_8 = np.percentile(pca_8.components_.T, args.lower_percentile)
upper_bound_8 = np.percentile(pca_8.components_.T, args.upper_percentile)
pca_9.fit(id_feats_9.cpu())
ash_bound_9 = np.percentile(pca_9.components_.T, args.ash_percentile)
lower_bound_9 = np.percentile(pca_9.components_.T, args.lower_percentile)
upper_bound_9 = np.percentile(pca_9.components_.T, args.upper_percentile)
if args.num_clusters > 10:
    pca_10.fit(id_feats_10.cpu())
    pca_11.fit(id_feats_11.cpu())
    pca_12.fit(id_feats_12.cpu())
    pca_13.fit(id_feats_13.cpu())
    pca_14.fit(id_feats_14.cpu())
    pca_15.fit(id_feats_15.cpu())
    pca_16.fit(id_feats_16.cpu())
    pca_17.fit(id_feats_17.cpu())
    pca_18.fit(id_feats_18.cpu())
    pca_19.fit(id_feats_19.cpu())

# pdb.set_trace()

dist_matrix = ((ood_feats.cpu().unsqueeze(1) - torch.Tensor(centroids).unsqueeze(0)) ** 2).mean(-1).sqrt()
ood_cluster_labels = torch.argmin(dist_matrix, dim=1)
ood_feats_0 = ood_feats[ood_cluster_labels == 0]
ood_feats_1 = ood_feats[ood_cluster_labels == 1]
ood_feats_2 = ood_feats[ood_cluster_labels == 2]
ood_feats_3 = ood_feats[ood_cluster_labels == 3]
ood_feats_4 = ood_feats[ood_cluster_labels == 4]
ood_feats_5 = ood_feats[ood_cluster_labels == 5]
ood_feats_6 = ood_feats[ood_cluster_labels == 6]
ood_feats_7 = ood_feats[ood_cluster_labels == 7]
ood_feats_8 = ood_feats[ood_cluster_labels == 8]
ood_feats_9 = ood_feats[ood_cluster_labels == 9]
if args.num_clusters > 10:
    ood_feats_10 = ood_feats[ood_cluster_labels == 10]
    ood_feats_11 = ood_feats[ood_cluster_labels == 11]
    ood_feats_12 = ood_feats[ood_cluster_labels == 12]
    ood_feats_13 = ood_feats[ood_cluster_labels == 13]
    ood_feats_14 = ood_feats[ood_cluster_labels == 14]
    ood_feats_15 = ood_feats[ood_cluster_labels == 15]
    ood_feats_16 = ood_feats[ood_cluster_labels == 16]
    ood_feats_17 = ood_feats[ood_cluster_labels == 17]
    ood_feats_18 = ood_feats[ood_cluster_labels == 18]
    ood_feats_19 = ood_feats[ood_cluster_labels == 19]

# def react_ash_transforms(m_trans, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a):
# def react_ash_transforms(m_trans, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a):
def react_ash_transforms(m_trans, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a):
    m_trans[:, 0] = react(m_trans[:, 0], clamp0_r)
    m_trans[:, 1] = react(m_trans[:, 1], clamp1_r)
    m_trans[:, 2] = react(m_trans[:, 2], clamp2_r)
    m_trans[:, 3] = react(m_trans[:, 3], clamp3_r)
    m_trans[:, 4] = react(m_trans[:, 4], clamp4_r)
    m_trans[:, 5] = react(m_trans[:, 5], clamp5_r)
    m_trans[:, 6] = react(m_trans[:, 6], clamp6_r)
    m_trans[:, 7] = react(m_trans[:, 7], clamp7_r)
    m_trans[:, 8] = react(m_trans[:, 8], clamp8_r)
    m_trans[:, 9] = react(m_trans[:, 9], clamp9_r)
    if args.num_component > 10:
        m_trans[:, 10] = react(m_trans[:, 10], clamp10_r)
        m_trans[:, 11] = react(m_trans[:, 11], clamp11_r)
        m_trans[:, 12] = react(m_trans[:, 12], clamp12_r)
        m_trans[:, 13] = react(m_trans[:, 13], clamp13_r)
        m_trans[:, 14] = react(m_trans[:, 14], clamp14_r)
        m_trans[:, 15] = react(m_trans[:, 15], clamp15_r)
        m_trans[:, 16] = react(m_trans[:, 16], clamp16_r)
        m_trans[:, 17] = react(m_trans[:, 17], clamp17_r)
        m_trans[:, 18] = react(m_trans[:, 18], clamp18_r)
        m_trans[:, 19] = react(m_trans[:, 19], clamp19_r)
    if args.num_component > 20:
        m_trans[:, 20] = react(m_trans[:, 20], clamp20_r)
        m_trans[:, 21] = react(m_trans[:, 21], clamp21_r)
        m_trans[:, 22] = react(m_trans[:, 22], clamp22_r)
        m_trans[:, 23] = react(m_trans[:, 23], clamp23_r)
        m_trans[:, 24] = react(m_trans[:, 24], clamp24_r)
        m_trans[:, 25] = react(m_trans[:, 25], clamp25_r)
        m_trans[:, 26] = react(m_trans[:, 26], clamp26_r)
        m_trans[:, 27] = react(m_trans[:, 27], clamp27_r)
        m_trans[:, 28] = react(m_trans[:, 28], clamp28_r)
        m_trans[:, 29] = react(m_trans[:, 29], clamp29_r)
        m_trans[:, 30] = react(m_trans[:, 30], clamp30_r)
        m_trans[:, 31] = react(m_trans[:, 31], clamp31_r)
        m_trans[:, 32] = react(m_trans[:, 32], clamp32_r)
        m_trans[:, 33] = react(m_trans[:, 33], clamp33_r)
        m_trans[:, 34] = react(m_trans[:, 34], clamp34_r)

    m_trans[:, 0] = ash_s_thre(m_trans[:, 0], clamp0_a)
    m_trans[:, 1] = ash_s_thre(m_trans[:, 1], clamp1_a)
    m_trans[:, 2] = ash_s_thre(m_trans[:, 2], clamp2_a)
    m_trans[:, 3] = ash_s_thre(m_trans[:, 3], clamp3_a)
    m_trans[:, 4] = ash_s_thre(m_trans[:, 4], clamp4_a)
    m_trans[:, 5] = ash_s_thre(m_trans[:, 5], clamp5_a)
    m_trans[:, 6] = ash_s_thre(m_trans[:, 6], clamp6_a)
    m_trans[:, 7] = ash_s_thre(m_trans[:, 7], clamp7_a)
    m_trans[:, 8] = ash_s_thre(m_trans[:, 8], clamp8_a)
    m_trans[:, 9] = ash_s_thre(m_trans[:, 9], clamp9_a)
    if args.num_component > 10:
        m_trans[:, 10] = ash_s_thre(m_trans[:, 10], clamp10_a)
        m_trans[:, 11] = ash_s_thre(m_trans[:, 11], clamp11_a)
        m_trans[:, 12] = ash_s_thre(m_trans[:, 12], clamp12_a)
        m_trans[:, 13] = ash_s_thre(m_trans[:, 13], clamp13_a)
        m_trans[:, 14] = ash_s_thre(m_trans[:, 14], clamp14_a)
        m_trans[:, 15] = ash_s_thre(m_trans[:, 15], clamp15_a)
        m_trans[:, 16] = ash_s_thre(m_trans[:, 16], clamp16_a)
        m_trans[:, 17] = ash_s_thre(m_trans[:, 17], clamp17_a)
        m_trans[:, 18] = ash_s_thre(m_trans[:, 18], clamp18_a)
        m_trans[:, 19] = ash_s_thre(m_trans[:, 19], clamp19_a)
    if args.num_component > 20:
        m_trans[:, 20] = ash_s_thre(m_trans[:, 20], clamp20_a)
        m_trans[:, 21] = ash_s_thre(m_trans[:, 21], clamp21_a)
        m_trans[:, 22] = ash_s_thre(m_trans[:, 22], clamp22_a)
        m_trans[:, 23] = ash_s_thre(m_trans[:, 23], clamp23_a)
        m_trans[:, 24] = ash_s_thre(m_trans[:, 24], clamp24_a)
        m_trans[:, 25] = ash_s_thre(m_trans[:, 25], clamp25_a)
        m_trans[:, 26] = ash_s_thre(m_trans[:, 26], clamp26_a)
        m_trans[:, 27] = ash_s_thre(m_trans[:, 27], clamp27_a)
        m_trans[:, 28] = ash_s_thre(m_trans[:, 28], clamp28_a)
        m_trans[:, 29] = ash_s_thre(m_trans[:, 29], clamp29_a)
        m_trans[:, 30] = ash_s_thre(m_trans[:, 30], clamp30_a)
        m_trans[:, 31] = ash_s_thre(m_trans[:, 31], clamp31_a)
        m_trans[:, 32] = ash_s_thre(m_trans[:, 32], clamp32_a)
        m_trans[:, 33] = ash_s_thre(m_trans[:, 33], clamp33_a)
        m_trans[:, 34] = ash_s_thre(m_trans[:, 34], clamp34_a)
    return m_trans

# def eval_datasets(clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a):
# def eval_datasets(clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a):
def eval_datasets(clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a):
    global id_feats_0, id_feats_1, id_feats_2, id_feats_3, id_feats_4, id_feats_5, id_feats_6, id_feats_7, id_feats_8, id_feats_9, final_fpr
    global ood_feats_0, ood_feats_1, ood_feats_2, ood_feats_3, ood_feats_4, ood_feats_5, ood_feats_6, ood_feats_7, ood_feats_8, ood_feats_9

    m_id_feats_0 = torch.clone(id_feats_0)
    m_id_feats_1 = torch.clone(id_feats_1)
    m_id_feats_2 = torch.clone(id_feats_2)
    m_id_feats_3 = torch.clone(id_feats_3)
    m_id_feats_4 = torch.clone(id_feats_4)
    m_id_feats_5 = torch.clone(id_feats_5)
    m_id_feats_6 = torch.clone(id_feats_6)
    m_id_feats_7 = torch.clone(id_feats_7)
    m_id_feats_8 = torch.clone(id_feats_8)
    m_id_feats_9 = torch.clone(id_feats_9)
    m_ood_feats_0 = torch.clone(ood_feats_0)
    m_ood_feats_1 = torch.clone(ood_feats_1)
    m_ood_feats_2 = torch.clone(ood_feats_2)
    m_ood_feats_3 = torch.clone(ood_feats_3)
    m_ood_feats_4 = torch.clone(ood_feats_4)
    m_ood_feats_5 = torch.clone(ood_feats_5)
    m_ood_feats_6 = torch.clone(ood_feats_6)
    m_ood_feats_7 = torch.clone(ood_feats_7)
    m_ood_feats_8 = torch.clone(ood_feats_8)
    m_ood_feats_9 = torch.clone(ood_feats_9)
    if args.num_clusters > 10:
        m_id_feats_10 = torch.clone(id_feats_10)
        m_id_feats_11 = torch.clone(id_feats_11)
        m_id_feats_12 = torch.clone(id_feats_12)
        m_id_feats_13 = torch.clone(id_feats_13)
        m_id_feats_14 = torch.clone(id_feats_14)
        m_id_feats_15 = torch.clone(id_feats_15)
        m_id_feats_16 = torch.clone(id_feats_16)
        m_id_feats_17 = torch.clone(id_feats_17)
        m_id_feats_18 = torch.clone(id_feats_18)
        m_id_feats_19 = torch.clone(id_feats_19)
        m_ood_feats_10 = torch.clone(ood_feats_10)
        m_ood_feats_11 = torch.clone(ood_feats_11)
        m_ood_feats_12 = torch.clone(ood_feats_12)
        m_ood_feats_13 = torch.clone(ood_feats_13)
        m_ood_feats_14 = torch.clone(ood_feats_14)
        m_ood_feats_15 = torch.clone(ood_feats_15)
        m_ood_feats_16 = torch.clone(ood_feats_16)
        m_ood_feats_17 = torch.clone(ood_feats_17)
        m_ood_feats_18 = torch.clone(ood_feats_18)
        m_ood_feats_19 = torch.clone(ood_feats_19)

    m_id_error_0 = m_id_feats_0 - (torch.Tensor(pca_0.transform(m_id_feats_0.cpu())).mm(torch.Tensor(pca_0.components_)).cuda() + id_feats_0.mean(0))
    m_id_error_1 = m_id_feats_1 - (torch.Tensor(pca_1.transform(m_id_feats_1.cpu())).mm(torch.Tensor(pca_1.components_)).cuda() + id_feats_1.mean(0))
    m_id_error_2 = m_id_feats_2 - (torch.Tensor(pca_2.transform(m_id_feats_2.cpu())).mm(torch.Tensor(pca_2.components_)).cuda() + id_feats_2.mean(0))
    m_id_error_3 = m_id_feats_3 - (torch.Tensor(pca_3.transform(m_id_feats_3.cpu())).mm(torch.Tensor(pca_3.components_)).cuda() + id_feats_3.mean(0))
    m_id_error_4 = m_id_feats_4 - (torch.Tensor(pca_4.transform(m_id_feats_4.cpu())).mm(torch.Tensor(pca_4.components_)).cuda() + id_feats_4.mean(0))
    m_id_error_5 = m_id_feats_5 - (torch.Tensor(pca_5.transform(m_id_feats_5.cpu())).mm(torch.Tensor(pca_5.components_)).cuda() + id_feats_5.mean(0))
    m_id_error_6 = m_id_feats_6 - (torch.Tensor(pca_6.transform(m_id_feats_6.cpu())).mm(torch.Tensor(pca_6.components_)).cuda() + id_feats_6.mean(0))
    m_id_error_7 = m_id_feats_7 - (torch.Tensor(pca_7.transform(m_id_feats_7.cpu())).mm(torch.Tensor(pca_7.components_)).cuda() + id_feats_7.mean(0))
    m_id_error_8 = m_id_feats_8 - (torch.Tensor(pca_8.transform(m_id_feats_8.cpu())).mm(torch.Tensor(pca_8.components_)).cuda() + id_feats_8.mean(0))
    m_id_error_9 = m_id_feats_9 - (torch.Tensor(pca_9.transform(m_id_feats_9.cpu())).mm(torch.Tensor(pca_9.components_)).cuda() + id_feats_9.mean(0))
    m_ood_error_0 = m_ood_feats_0 - (torch.Tensor(pca_0.transform(m_ood_feats_0.cpu())).mm(torch.Tensor(pca_0.components_)).cuda() + id_feats_0.mean(0))
    m_ood_error_1 = m_ood_feats_1 - (torch.Tensor(pca_1.transform(m_ood_feats_1.cpu())).mm(torch.Tensor(pca_1.components_)).cuda() + id_feats_1.mean(0))
    m_ood_error_2 = m_ood_feats_2 - (torch.Tensor(pca_2.transform(m_ood_feats_2.cpu())).mm(torch.Tensor(pca_2.components_)).cuda() + id_feats_2.mean(0))
    m_ood_error_3 = m_ood_feats_3 - (torch.Tensor(pca_3.transform(m_ood_feats_3.cpu())).mm(torch.Tensor(pca_3.components_)).cuda() + id_feats_3.mean(0))
    m_ood_error_4 = m_ood_feats_4 - (torch.Tensor(pca_4.transform(m_ood_feats_4.cpu())).mm(torch.Tensor(pca_4.components_)).cuda() + id_feats_4.mean(0))
    m_ood_error_5 = m_ood_feats_5 - (torch.Tensor(pca_5.transform(m_ood_feats_5.cpu())).mm(torch.Tensor(pca_5.components_)).cuda() + id_feats_5.mean(0))
    m_ood_error_6 = m_ood_feats_6 - (torch.Tensor(pca_6.transform(m_ood_feats_6.cpu())).mm(torch.Tensor(pca_6.components_)).cuda() + id_feats_6.mean(0))
    m_ood_error_7 = m_ood_feats_7 - (torch.Tensor(pca_7.transform(m_ood_feats_7.cpu())).mm(torch.Tensor(pca_7.components_)).cuda() + id_feats_7.mean(0))
    m_ood_error_8 = m_ood_feats_8 - (torch.Tensor(pca_8.transform(m_ood_feats_8.cpu())).mm(torch.Tensor(pca_8.components_)).cuda() + id_feats_8.mean(0))
    m_ood_error_9 = m_ood_feats_9 - (torch.Tensor(pca_9.transform(m_ood_feats_9.cpu())).mm(torch.Tensor(pca_9.components_)).cuda() + id_feats_9.mean(0))
    if args.num_clusters > 10:
        m_id_error_10 = m_id_feats_10 - (torch.Tensor(pca_0.transform(m_id_feats_10.cpu())).mm(torch.Tensor(pca_10.components_)).cuda() + id_feats_10.mean(0))
        m_id_error_11 = m_id_feats_11 - (torch.Tensor(pca_1.transform(m_id_feats_11.cpu())).mm(torch.Tensor(pca_11.components_)).cuda() + id_feats_11.mean(0))
        m_id_error_12 = m_id_feats_12 - (torch.Tensor(pca_2.transform(m_id_feats_12.cpu())).mm(torch.Tensor(pca_12.components_)).cuda() + id_feats_12.mean(0))
        m_id_error_13 = m_id_feats_13 - (torch.Tensor(pca_3.transform(m_id_feats_13.cpu())).mm(torch.Tensor(pca_13.components_)).cuda() + id_feats_13.mean(0))
        m_id_error_14 = m_id_feats_14 - (torch.Tensor(pca_4.transform(m_id_feats_14.cpu())).mm(torch.Tensor(pca_14.components_)).cuda() + id_feats_14.mean(0))
        m_id_error_15 = m_id_feats_15 - (torch.Tensor(pca_5.transform(m_id_feats_15.cpu())).mm(torch.Tensor(pca_15.components_)).cuda() + id_feats_15.mean(0))
        m_id_error_16 = m_id_feats_16 - (torch.Tensor(pca_6.transform(m_id_feats_16.cpu())).mm(torch.Tensor(pca_16.components_)).cuda() + id_feats_16.mean(0))
        m_id_error_17 = m_id_feats_17 - (torch.Tensor(pca_7.transform(m_id_feats_17.cpu())).mm(torch.Tensor(pca_17.components_)).cuda() + id_feats_17.mean(0))
        m_id_error_18 = m_id_feats_18 - (torch.Tensor(pca_8.transform(m_id_feats_18.cpu())).mm(torch.Tensor(pca_18.components_)).cuda() + id_feats_18.mean(0))
        m_id_error_19 = m_id_feats_19 - (torch.Tensor(pca_9.transform(m_id_feats_19.cpu())).mm(torch.Tensor(pca_19.components_)).cuda() + id_feats_19.mean(0))
        m_ood_error_10 = m_ood_feats_10 - (torch.Tensor(pca_0.transform(m_ood_feats_10.cpu())).mm(torch.Tensor(pca_10.components_)).cuda() + id_feats_10.mean(0))
        m_ood_error_11 = m_ood_feats_11 - (torch.Tensor(pca_1.transform(m_ood_feats_11.cpu())).mm(torch.Tensor(pca_11.components_)).cuda() + id_feats_11.mean(0))
        m_ood_error_12 = m_ood_feats_12 - (torch.Tensor(pca_2.transform(m_ood_feats_12.cpu())).mm(torch.Tensor(pca_12.components_)).cuda() + id_feats_12.mean(0))
        m_ood_error_13 = m_ood_feats_13 - (torch.Tensor(pca_3.transform(m_ood_feats_13.cpu())).mm(torch.Tensor(pca_13.components_)).cuda() + id_feats_13.mean(0))
        m_ood_error_14 = m_ood_feats_14 - (torch.Tensor(pca_4.transform(m_ood_feats_14.cpu())).mm(torch.Tensor(pca_14.components_)).cuda() + id_feats_14.mean(0))
        m_ood_error_15 = m_ood_feats_15 - (torch.Tensor(pca_5.transform(m_ood_feats_15.cpu())).mm(torch.Tensor(pca_15.components_)).cuda() + id_feats_15.mean(0))
        m_ood_error_16 = m_ood_feats_16 - (torch.Tensor(pca_6.transform(m_ood_feats_16.cpu())).mm(torch.Tensor(pca_16.components_)).cuda() + id_feats_16.mean(0))
        m_ood_error_17 = m_ood_feats_17 - (torch.Tensor(pca_7.transform(m_ood_feats_17.cpu())).mm(torch.Tensor(pca_17.components_)).cuda() + id_feats_17.mean(0))
        m_ood_error_18 = m_ood_feats_18 - (torch.Tensor(pca_8.transform(m_ood_feats_18.cpu())).mm(torch.Tensor(pca_18.components_)).cuda() + id_feats_18.mean(0))
        m_ood_error_19 = m_ood_feats_19 - (torch.Tensor(pca_9.transform(m_ood_feats_19.cpu())).mm(torch.Tensor(pca_19.components_)).cuda() + id_feats_19.mean(0))

    m_trans_0 = torch.clone(torch.Tensor(pca_0.components_.T))
    m_trans_1 = torch.clone(torch.Tensor(pca_1.components_.T))
    m_trans_2 = torch.clone(torch.Tensor(pca_2.components_.T))
    m_trans_3 = torch.clone(torch.Tensor(pca_3.components_.T))
    m_trans_4 = torch.clone(torch.Tensor(pca_4.components_.T))
    m_trans_5 = torch.clone(torch.Tensor(pca_5.components_.T))
    m_trans_6 = torch.clone(torch.Tensor(pca_6.components_.T))
    m_trans_7 = torch.clone(torch.Tensor(pca_7.components_.T))
    m_trans_8 = torch.clone(torch.Tensor(pca_8.components_.T))
    m_trans_9 = torch.clone(torch.Tensor(pca_9.components_.T))
    if args.num_clusters > 10:
        m_trans_10 = torch.clone(torch.Tensor(pca_10.components_.T))
        m_trans_11 = torch.clone(torch.Tensor(pca_11.components_.T))
        m_trans_12 = torch.clone(torch.Tensor(pca_12.components_.T))
        m_trans_13 = torch.clone(torch.Tensor(pca_13.components_.T))
        m_trans_14 = torch.clone(torch.Tensor(pca_14.components_.T))
        m_trans_15 = torch.clone(torch.Tensor(pca_15.components_.T))
        m_trans_16 = torch.clone(torch.Tensor(pca_16.components_.T))
        m_trans_17 = torch.clone(torch.Tensor(pca_17.components_.T))
        m_trans_18 = torch.clone(torch.Tensor(pca_18.components_.T))
        m_trans_19 = torch.clone(torch.Tensor(pca_19.components_.T))

    # clamping
    # m_trans_0 = react_ash_transforms(m_trans_0, clamp0_1, clamp1_1, clamp2_1, clamp3_1, clamp4_1, clamp5_1, clamp6_1, clamp7_1, clamp8_1, clamp9_1, clamp10_1, clamp11_1, clamp12_1, clamp13_1, clamp14_1, clamp15_1, clamp16_1, clamp17_1, clamp18_1, clamp19_1)
    # m_trans_1 = react_ash_transforms(m_trans_1, clamp0_2, clamp1_2, clamp2_2, clamp3_2, clamp4_2, clamp5_2, clamp6_2, clamp7_2, clamp8_2, clamp9_2, clamp10_2, clamp11_2, clamp12_2, clamp13_2, clamp14_2, clamp15_2, clamp16_2, clamp17_2, clamp18_2, clamp19_2)
    # m_trans_2 = react_ash_transforms(m_trans_2, clamp0_3, clamp1_3, clamp2_3, clamp3_3, clamp4_3, clamp5_3, clamp6_3, clamp7_3, clamp8_3, clamp9_3, clamp10_3, clamp11_3, clamp12_3, clamp13_3, clamp14_3, clamp15_3, clamp16_3, clamp17_3, clamp18_3, clamp19_3)
    # m_trans_3 = react_ash_transforms(m_trans_3, clamp0_4, clamp1_4, clamp2_4, clamp3_4, clamp4_4, clamp5_4, clamp6_4, clamp7_4, clamp8_4, clamp9_4, clamp10_4, clamp11_4, clamp12_4, clamp13_4, clamp14_4, clamp15_4, clamp16_4, clamp17_4, clamp18_4, clamp19_4)
    # m_trans_4 = react_ash_transforms(m_trans_4, clamp0_5, clamp1_5, clamp2_5, clamp3_5, clamp4_5, clamp5_5, clamp6_5, clamp7_5, clamp8_5, clamp9_5, clamp10_5, clamp11_5, clamp12_5, clamp13_5, clamp14_5, clamp15_5, clamp16_5, clamp17_5, clamp18_5, clamp19_5)
    # m_trans_5 = react_ash_transforms(m_trans_5, clamp0_6, clamp1_6, clamp2_6, clamp3_6, clamp4_6, clamp5_6, clamp6_6, clamp7_6, clamp8_6, clamp9_6, clamp10_6, clamp11_6, clamp12_6, clamp13_6, clamp14_6, clamp15_6, clamp16_6, clamp17_6, clamp18_6, clamp19_6)
    # m_trans_6 = react_ash_transforms(m_trans_6, clamp0_7, clamp1_7, clamp2_7, clamp3_7, clamp4_7, clamp5_7, clamp6_7, clamp7_7, clamp8_7, clamp9_7, clamp10_7, clamp11_7, clamp12_7, clamp13_7, clamp14_7, clamp15_7, clamp16_7, clamp17_7, clamp18_7, clamp19_7)
    # m_trans_7 = react_ash_transforms(m_trans_7, clamp0_8, clamp1_8, clamp2_8, clamp3_8, clamp4_8, clamp5_8, clamp6_8, clamp7_8, clamp8_8, clamp9_8, clamp10_8, clamp11_8, clamp12_8, clamp13_8, clamp14_8, clamp15_8, clamp16_8, clamp17_8, clamp18_8, clamp19_8)
    # m_trans_8 = react_ash_transforms(m_trans_8, clamp0_9, clamp1_9, clamp2_9, clamp3_9, clamp4_9, clamp5_9, clamp6_9, clamp7_9, clamp8_9, clamp9_9, clamp10_9, clamp11_9, clamp12_9, clamp13_9, clamp14_9, clamp15_9, clamp16_9, clamp17_9, clamp18_9, clamp19_9)
    # m_trans_9 = react_ash_transforms(m_trans_9, clamp0_10, clamp1_10, clamp2_10, clamp3_10, clamp4_10, clamp5_10, clamp6_10, clamp7_10, clamp8_10, clamp9_10, clamp10_10, clamp11_10, clamp12_10, clamp13_10, clamp14_10, clamp15_10, clamp16_10, clamp17_10, clamp18_10, clamp19_10)
    
    if args.num_component == 10:
        m_trans_0 = react_ash_transforms(m_trans_0, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_1 = react_ash_transforms(m_trans_1, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_2 = react_ash_transforms(m_trans_2, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_3 = react_ash_transforms(m_trans_3, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_4 = react_ash_transforms(m_trans_4, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_5 = react_ash_transforms(m_trans_5, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_6 = react_ash_transforms(m_trans_6, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_7 = react_ash_transforms(m_trans_7, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_8 = react_ash_transforms(m_trans_8, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        m_trans_9 = react_ash_transforms(m_trans_9, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
        if args.num_clusters > 10:
            m_trans_10 = react_ash_transforms(m_trans_10, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_11 = react_ash_transforms(m_trans_11, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_12 = react_ash_transforms(m_trans_12, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_13 = react_ash_transforms(m_trans_13, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_14 = react_ash_transforms(m_trans_14, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_15 = react_ash_transforms(m_trans_15, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_16 = react_ash_transforms(m_trans_16, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_17 = react_ash_transforms(m_trans_17, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_18 = react_ash_transforms(m_trans_18, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
            m_trans_19 = react_ash_transforms(m_trans_19, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a)
    elif args.num_component == 20:
        m_trans_0 = react_ash_transforms(m_trans_0, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_1 = react_ash_transforms(m_trans_1, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_2 = react_ash_transforms(m_trans_2, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_3 = react_ash_transforms(m_trans_3, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_4 = react_ash_transforms(m_trans_4, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_5 = react_ash_transforms(m_trans_5, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_6 = react_ash_transforms(m_trans_6, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_7 = react_ash_transforms(m_trans_7, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_8 = react_ash_transforms(m_trans_8, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        m_trans_9 = react_ash_transforms(m_trans_9, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
        if args.num_clusters > 10:
            m_trans_10 = react_ash_transforms(m_trans_10, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_11 = react_ash_transforms(m_trans_11, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_12 = react_ash_transforms(m_trans_12, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_13 = react_ash_transforms(m_trans_13, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_14 = react_ash_transforms(m_trans_14, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_15 = react_ash_transforms(m_trans_15, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_16 = react_ash_transforms(m_trans_16, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_17 = react_ash_transforms(m_trans_17, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_18 = react_ash_transforms(m_trans_18, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
            m_trans_19 = react_ash_transforms(m_trans_19, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a)
    elif args.num_component == 35:
        m_trans_0 = react_ash_transforms(m_trans_0, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_1 = react_ash_transforms(m_trans_1, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_2 = react_ash_transforms(m_trans_2, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_3 = react_ash_transforms(m_trans_3, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_4 = react_ash_transforms(m_trans_4, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_5 = react_ash_transforms(m_trans_5, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_6 = react_ash_transforms(m_trans_6, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_7 = react_ash_transforms(m_trans_7, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_8 = react_ash_transforms(m_trans_8, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        m_trans_9 = react_ash_transforms(m_trans_9, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
        if args.num_clusters > 10:
            m_trans_10 = react_ash_transforms(m_trans_10, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_11 = react_ash_transforms(m_trans_11, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_12 = react_ash_transforms(m_trans_12, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_13 = react_ash_transforms(m_trans_13, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_14 = react_ash_transforms(m_trans_14, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_15 = react_ash_transforms(m_trans_15, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_16 = react_ash_transforms(m_trans_16, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_17 = react_ash_transforms(m_trans_17, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_18 = react_ash_transforms(m_trans_18, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)
            m_trans_19 = react_ash_transforms(m_trans_19, clamp0_r, clamp1_r, clamp2_r, clamp3_r, clamp4_r, clamp5_r, clamp6_r, clamp7_r, clamp8_r, clamp9_r, clamp10_r, clamp11_r, clamp12_r, clamp13_r, clamp14_r, clamp15_r, clamp16_r, clamp17_r, clamp18_r, clamp19_r, clamp20_r, clamp21_r, clamp22_r, clamp23_r, clamp24_r, clamp25_r, clamp26_r, clamp27_r, clamp28_r, clamp29_r, clamp30_r, clamp31_r, clamp32_r, clamp33_r, clamp34_r, clamp0_a, clamp1_a, clamp2_a, clamp3_a, clamp4_a, clamp5_a, clamp6_a, clamp7_a, clamp8_a, clamp9_a, clamp10_a, clamp11_a, clamp12_a, clamp13_a, clamp14_a, clamp15_a, clamp16_a, clamp17_a, clamp18_a, clamp19_a, clamp20_a, clamp21_a, clamp22_a, clamp23_a, clamp24_a, clamp25_a, clamp26_a, clamp27_a, clamp28_a, clamp29_a, clamp30_a, clamp31_a, clamp32_a, clamp33_a, clamp34_a)

    m_id_feats_0 = torch.Tensor(pca_0.transform(m_id_feats_0.cpu())).mm(m_trans_0.T).cuda() + id_feats_0.mean(0) + m_id_error_0
    m_id_feats_1 = torch.Tensor(pca_1.transform(m_id_feats_1.cpu())).mm(m_trans_1.T).cuda() + id_feats_1.mean(0) + m_id_error_1
    m_id_feats_2 = torch.Tensor(pca_2.transform(m_id_feats_2.cpu())).mm(m_trans_2.T).cuda() + id_feats_2.mean(0) + m_id_error_2
    m_id_feats_3 = torch.Tensor(pca_3.transform(m_id_feats_3.cpu())).mm(m_trans_3.T).cuda() + id_feats_3.mean(0) + m_id_error_3
    m_id_feats_4 = torch.Tensor(pca_4.transform(m_id_feats_4.cpu())).mm(m_trans_4.T).cuda() + id_feats_4.mean(0) + m_id_error_4
    m_id_feats_5 = torch.Tensor(pca_5.transform(m_id_feats_5.cpu())).mm(m_trans_5.T).cuda() + id_feats_5.mean(0) + m_id_error_5
    m_id_feats_6 = torch.Tensor(pca_6.transform(m_id_feats_6.cpu())).mm(m_trans_6.T).cuda() + id_feats_6.mean(0) + m_id_error_6
    m_id_feats_7 = torch.Tensor(pca_7.transform(m_id_feats_7.cpu())).mm(m_trans_7.T).cuda() + id_feats_7.mean(0) + m_id_error_7
    m_id_feats_8 = torch.Tensor(pca_8.transform(m_id_feats_8.cpu())).mm(m_trans_8.T).cuda() + id_feats_8.mean(0) + m_id_error_8
    m_id_feats_9 = torch.Tensor(pca_9.transform(m_id_feats_9.cpu())).mm(m_trans_9.T).cuda() + id_feats_9.mean(0) + m_id_error_9
    m_ood_feats_0 = torch.Tensor(pca_0.transform(m_ood_feats_0.cpu())).mm(m_trans_0.T).cuda() + id_feats_0.mean(0) + m_ood_error_0
    m_ood_feats_1 = torch.Tensor(pca_1.transform(m_ood_feats_1.cpu())).mm(m_trans_1.T).cuda() + id_feats_1.mean(0) + m_ood_error_1
    m_ood_feats_2 = torch.Tensor(pca_2.transform(m_ood_feats_2.cpu())).mm(m_trans_2.T).cuda() + id_feats_2.mean(0) + m_ood_error_2
    m_ood_feats_3 = torch.Tensor(pca_3.transform(m_ood_feats_3.cpu())).mm(m_trans_3.T).cuda() + id_feats_3.mean(0) + m_ood_error_3
    m_ood_feats_4 = torch.Tensor(pca_4.transform(m_ood_feats_4.cpu())).mm(m_trans_4.T).cuda() + id_feats_4.mean(0) + m_ood_error_4
    m_ood_feats_5 = torch.Tensor(pca_5.transform(m_ood_feats_5.cpu())).mm(m_trans_5.T).cuda() + id_feats_5.mean(0) + m_ood_error_5
    m_ood_feats_6 = torch.Tensor(pca_6.transform(m_ood_feats_6.cpu())).mm(m_trans_6.T).cuda() + id_feats_6.mean(0) + m_ood_error_6
    m_ood_feats_7 = torch.Tensor(pca_7.transform(m_ood_feats_7.cpu())).mm(m_trans_7.T).cuda() + id_feats_7.mean(0) + m_ood_error_7
    m_ood_feats_8 = torch.Tensor(pca_8.transform(m_ood_feats_8.cpu())).mm(m_trans_8.T).cuda() + id_feats_8.mean(0) + m_ood_error_8
    m_ood_feats_9 = torch.Tensor(pca_9.transform(m_ood_feats_9.cpu())).mm(m_trans_9.T).cuda() + id_feats_9.mean(0) + m_ood_error_9
    if args.num_clusters > 10:
        m_id_feats_10 = torch.Tensor(pca_10.transform(m_id_feats_10.cpu())).mm(m_trans_10.T).cuda() + id_feats_10.mean(0) + m_id_error_10
        m_id_feats_11 = torch.Tensor(pca_11.transform(m_id_feats_11.cpu())).mm(m_trans_11.T).cuda() + id_feats_11.mean(0) + m_id_error_11
        m_id_feats_12 = torch.Tensor(pca_12.transform(m_id_feats_12.cpu())).mm(m_trans_12.T).cuda() + id_feats_12.mean(0) + m_id_error_12
        m_id_feats_13 = torch.Tensor(pca_13.transform(m_id_feats_13.cpu())).mm(m_trans_13.T).cuda() + id_feats_13.mean(0) + m_id_error_13
        m_id_feats_14 = torch.Tensor(pca_14.transform(m_id_feats_14.cpu())).mm(m_trans_14.T).cuda() + id_feats_14.mean(0) + m_id_error_14
        m_id_feats_15 = torch.Tensor(pca_15.transform(m_id_feats_15.cpu())).mm(m_trans_15.T).cuda() + id_feats_15.mean(0) + m_id_error_15
        m_id_feats_16 = torch.Tensor(pca_16.transform(m_id_feats_16.cpu())).mm(m_trans_16.T).cuda() + id_feats_16.mean(0) + m_id_error_16
        m_id_feats_17 = torch.Tensor(pca_17.transform(m_id_feats_17.cpu())).mm(m_trans_17.T).cuda() + id_feats_17.mean(0) + m_id_error_17
        m_id_feats_18 = torch.Tensor(pca_18.transform(m_id_feats_18.cpu())).mm(m_trans_18.T).cuda() + id_feats_18.mean(0) + m_id_error_18
        m_id_feats_19 = torch.Tensor(pca_19.transform(m_id_feats_19.cpu())).mm(m_trans_19.T).cuda() + id_feats_19.mean(0) + m_id_error_19
        m_ood_feats_10 = torch.Tensor(pca_10.transform(m_ood_feats_10.cpu())).mm(m_trans_10.T).cuda() + id_feats_10.mean(0) + m_ood_error_10
        m_ood_feats_11 = torch.Tensor(pca_11.transform(m_ood_feats_11.cpu())).mm(m_trans_11.T).cuda() + id_feats_11.mean(0) + m_ood_error_11
        m_ood_feats_12 = torch.Tensor(pca_12.transform(m_ood_feats_12.cpu())).mm(m_trans_12.T).cuda() + id_feats_12.mean(0) + m_ood_error_12
        m_ood_feats_13 = torch.Tensor(pca_13.transform(m_ood_feats_13.cpu())).mm(m_trans_13.T).cuda() + id_feats_13.mean(0) + m_ood_error_13
        m_ood_feats_14 = torch.Tensor(pca_14.transform(m_ood_feats_14.cpu())).mm(m_trans_14.T).cuda() + id_feats_14.mean(0) + m_ood_error_14
        m_ood_feats_15 = torch.Tensor(pca_15.transform(m_ood_feats_15.cpu())).mm(m_trans_15.T).cuda() + id_feats_15.mean(0) + m_ood_error_15
        m_ood_feats_16 = torch.Tensor(pca_16.transform(m_ood_feats_16.cpu())).mm(m_trans_16.T).cuda() + id_feats_16.mean(0) + m_ood_error_16
        m_ood_feats_17 = torch.Tensor(pca_17.transform(m_ood_feats_17.cpu())).mm(m_trans_17.T).cuda() + id_feats_17.mean(0) + m_ood_error_17
        m_ood_feats_18 = torch.Tensor(pca_18.transform(m_ood_feats_18.cpu())).mm(m_trans_18.T).cuda() + id_feats_18.mean(0) + m_ood_error_18
        m_ood_feats_19 = torch.Tensor(pca_19.transform(m_ood_feats_19.cpu())).mm(m_trans_19.T).cuda() + id_feats_19.mean(0) + m_ood_error_19

    if args.num_clusters == 10:
        m_id_feats = torch.cat((m_id_feats_0, m_id_feats_1, m_id_feats_2, m_id_feats_3, m_id_feats_4, m_id_feats_5, m_id_feats_6, m_id_feats_7, m_id_feats_8, m_id_feats_9), dim=0)
        m_ood_feats = torch.cat((m_ood_feats_0, m_ood_feats_1, m_ood_feats_2, m_ood_feats_3, m_ood_feats_4, m_ood_feats_5, m_ood_feats_6, m_ood_feats_7, m_ood_feats_8, m_ood_feats_9), dim=0)
    elif args.num_clusters == 20:
        m_id_feats = torch.cat((m_id_feats_0, m_id_feats_1, m_id_feats_2, m_id_feats_3, m_id_feats_4, m_id_feats_5, m_id_feats_6, m_id_feats_7, m_id_feats_8, m_id_feats_9, m_id_feats_10, m_id_feats_11, m_id_feats_12, m_id_feats_13, m_id_feats_14, m_id_feats_15, m_id_feats_16, m_id_feats_17, m_id_feats_18, m_id_feats_19), dim=0)
        m_ood_feats = torch.cat((m_ood_feats_0, m_ood_feats_1, m_ood_feats_2, m_ood_feats_3, m_ood_feats_4, m_ood_feats_5, m_ood_feats_6, m_ood_feats_7, m_ood_feats_8, m_ood_feats_9, m_ood_feats_10, m_ood_feats_11, m_ood_feats_12, m_ood_feats_13, m_ood_feats_14, m_ood_feats_15, m_ood_feats_16, m_ood_feats_17, m_ood_feats_18, m_ood_feats_19), dim=0)
    m_id_logits = model.fc(m_id_feats)
    m_ood_logits = model.fc(m_ood_feats)
    m_id_score =  - torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
    m_ood_score =  - torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()
    if torch.isnan(torch.Tensor(m_id_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood_score)).sum() != 0:
        return -1e9

    fpr, auroc, aupr = score_get_and_print_results(log, m_id_score, m_ood_score)
    if fpr < final_fpr:
        final_fpr = fpr
    wandb.log({"final_fpr": final_fpr})
    return auroc - fpr

ood_bayesian = BayesianOptimization(
    eval_datasets,
    {
    # 'clamp0_1': (lower_bound_0, upper_bound_0),
    # 'clamp1_1': (lower_bound_0, upper_bound_0),
    # 'clamp2_1': (lower_bound_0, upper_bound_0),
    # 'clamp3_1': (lower_bound_0, upper_bound_0),
    # 'clamp4_1': (lower_bound_0, upper_bound_0),
    # 'clamp5_1': (lower_bound_0, upper_bound_0),
    # 'clamp6_1': (lower_bound_0, upper_bound_0),
    # 'clamp7_1': (lower_bound_0, upper_bound_0),
    # 'clamp8_1': (lower_bound_0, upper_bound_0),
    # 'clamp9_1': (lower_bound_0, upper_bound_0),
    # 'clamp10_1': (-1, ash_bound_0),
    # 'clamp11_1': (-1, ash_bound_0),
    # 'clamp12_1': (-1, ash_bound_0),
    # 'clamp13_1': (-1, ash_bound_0),
    # 'clamp14_1': (-1, ash_bound_0),
    # 'clamp15_1': (-1, ash_bound_0),
    # 'clamp16_1': (-1, ash_bound_0),
    # 'clamp17_1': (-1, ash_bound_0),
    # 'clamp18_1': (-1, ash_bound_0),
    # 'clamp19_1': (-1, ash_bound_0),
    # 'clamp0_2': (lower_bound_1, upper_bound_1),
    # 'clamp1_2': (lower_bound_1, upper_bound_1),
    # 'clamp2_2': (lower_bound_1, upper_bound_1),
    # 'clamp3_2': (lower_bound_1, upper_bound_1),
    # 'clamp4_2': (lower_bound_1, upper_bound_1),
    # 'clamp5_2': (lower_bound_1, upper_bound_1),
    # 'clamp6_2': (lower_bound_1, upper_bound_1),
    # 'clamp7_2': (lower_bound_1, upper_bound_1),
    # 'clamp8_2': (lower_bound_1, upper_bound_1),
    # 'clamp9_2': (lower_bound_1, upper_bound_1),
    # 'clamp10_2': (-1, ash_bound_1),
    # 'clamp11_2': (-1, ash_bound_1),
    # 'clamp12_2': (-1, ash_bound_1),
    # 'clamp13_2': (-1, ash_bound_1),
    # 'clamp14_2': (-1, ash_bound_1),
    # 'clamp15_2': (-1, ash_bound_1),
    # 'clamp16_2': (-1, ash_bound_1),
    # 'clamp17_2': (-1, ash_bound_1),
    # 'clamp18_2': (-1, ash_bound_1),
    # 'clamp19_2': (-1, ash_bound_1),
    # 'clamp0_3': (lower_bound_2, upper_bound_2),
    # 'clamp1_3': (lower_bound_2, upper_bound_2),
    # 'clamp2_3': (lower_bound_2, upper_bound_2),
    # 'clamp3_3': (lower_bound_2, upper_bound_2),
    # 'clamp4_3': (lower_bound_2, upper_bound_2),
    # 'clamp5_3': (lower_bound_2, upper_bound_2),
    # 'clamp6_3': (lower_bound_2, upper_bound_2),
    # 'clamp7_3': (lower_bound_2, upper_bound_2),
    # 'clamp8_3': (lower_bound_2, upper_bound_2),
    # 'clamp9_3': (lower_bound_2, upper_bound_2),
    # 'clamp10_3': (-1, ash_bound_2),
    # 'clamp11_3': (-1, ash_bound_2),
    # 'clamp12_3': (-1, ash_bound_2),
    # 'clamp13_3': (-1, ash_bound_2),
    # 'clamp14_3': (-1, ash_bound_2),
    # 'clamp15_3': (-1, ash_bound_2),
    # 'clamp16_3': (-1, ash_bound_2),
    # 'clamp17_3': (-1, ash_bound_2),
    # 'clamp18_3': (-1, ash_bound_2),
    # 'clamp19_3': (-1, ash_bound_2),
    # 'clamp0_4': (lower_bound_3, upper_bound_3),
    # 'clamp1_4': (lower_bound_3, upper_bound_3),
    # 'clamp2_4': (lower_bound_3, upper_bound_3),
    # 'clamp3_4': (lower_bound_3, upper_bound_3),
    # 'clamp4_4': (lower_bound_3, upper_bound_3),
    # 'clamp5_4': (lower_bound_3, upper_bound_3),
    # 'clamp6_4': (lower_bound_3, upper_bound_3),
    # 'clamp7_4': (lower_bound_3, upper_bound_3),
    # 'clamp8_4': (lower_bound_3, upper_bound_3),
    # 'clamp9_4': (lower_bound_3, upper_bound_3),
    # 'clamp10_4': (-1, ash_bound_3),
    # 'clamp11_4': (-1, ash_bound_3),
    # 'clamp12_4': (-1, ash_bound_3),
    # 'clamp13_4': (-1, ash_bound_3),
    # 'clamp14_4': (-1, ash_bound_3),
    # 'clamp15_4': (-1, ash_bound_3),
    # 'clamp16_4': (-1, ash_bound_3),
    # 'clamp17_4': (-1, ash_bound_3),
    # 'clamp18_4': (-1, ash_bound_3),
    # 'clamp19_4': (-1, ash_bound_3),
    # 'clamp0_5': (lower_bound_4, upper_bound_4),
    # 'clamp1_5': (lower_bound_4, upper_bound_4),
    # 'clamp2_5': (lower_bound_4, upper_bound_4),
    # 'clamp3_5': (lower_bound_4, upper_bound_4),
    # 'clamp4_5': (lower_bound_4, upper_bound_4),
    # 'clamp5_5': (lower_bound_4, upper_bound_4),
    # 'clamp6_5': (lower_bound_4, upper_bound_4),
    # 'clamp7_5': (lower_bound_4, upper_bound_4),
    # 'clamp8_5': (lower_bound_4, upper_bound_4),
    # 'clamp9_5': (lower_bound_4, upper_bound_4),
    # 'clamp10_5': (-1, ash_bound_4),
    # 'clamp11_5': (-1, ash_bound_4),
    # 'clamp12_5': (-1, ash_bound_4),
    # 'clamp13_5': (-1, ash_bound_4),
    # 'clamp14_5': (-1, ash_bound_4),
    # 'clamp15_5': (-1, ash_bound_4),
    # 'clamp16_5': (-1, ash_bound_4),
    # 'clamp17_5': (-1, ash_bound_4),
    # 'clamp18_5': (-1, ash_bound_4),
    # 'clamp19_5': (-1, ash_bound_4),
    # 'clamp0_6': (lower_bound_5, upper_bound_5),
    # 'clamp1_6': (lower_bound_5, upper_bound_5),
    # 'clamp2_6': (lower_bound_5, upper_bound_5),
    # 'clamp3_6': (lower_bound_5, upper_bound_5),
    # 'clamp4_6': (lower_bound_5, upper_bound_5),
    # 'clamp5_6': (lower_bound_5, upper_bound_5),
    # 'clamp6_6': (lower_bound_5, upper_bound_5),
    # 'clamp7_6': (lower_bound_5, upper_bound_5),
    # 'clamp8_6': (lower_bound_5, upper_bound_5),
    # 'clamp9_6': (lower_bound_5, upper_bound_5),
    # 'clamp10_6': (-1, ash_bound_5),
    # 'clamp11_6': (-1, ash_bound_5),
    # 'clamp12_6': (-1, ash_bound_5),
    # 'clamp13_6': (-1, ash_bound_5),
    # 'clamp14_6': (-1, ash_bound_5),
    # 'clamp15_6': (-1, ash_bound_5),
    # 'clamp16_6': (-1, ash_bound_5),
    # 'clamp17_6': (-1, ash_bound_5),
    # 'clamp18_6': (-1, ash_bound_5),
    # 'clamp19_6': (-1, ash_bound_5),
    # 'clamp0_7': (lower_bound_6, upper_bound_6),
    # 'clamp1_7': (lower_bound_6, upper_bound_6),
    # 'clamp2_7': (lower_bound_6, upper_bound_6),
    # 'clamp3_7': (lower_bound_6, upper_bound_6),
    # 'clamp4_7': (lower_bound_6, upper_bound_6),
    # 'clamp5_7': (lower_bound_6, upper_bound_6),
    # 'clamp6_7': (lower_bound_6, upper_bound_6),
    # 'clamp7_7': (lower_bound_6, upper_bound_6),
    # 'clamp8_7': (lower_bound_6, upper_bound_6),
    # 'clamp9_7': (lower_bound_6, upper_bound_6),
    # 'clamp10_7': (-1, ash_bound_6),
    # 'clamp11_7': (-1, ash_bound_6),
    # 'clamp12_7': (-1, ash_bound_6),
    # 'clamp13_7': (-1, ash_bound_6),
    # 'clamp14_7': (-1, ash_bound_6),
    # 'clamp15_7': (-1, ash_bound_6),
    # 'clamp16_7': (-1, ash_bound_6),
    # 'clamp17_7': (-1, ash_bound_6),
    # 'clamp18_7': (-1, ash_bound_6),
    # 'clamp19_7': (-1, ash_bound_6),
    # 'clamp0_8': (lower_bound_7, upper_bound_7),
    # 'clamp1_8': (lower_bound_7, upper_bound_7),
    # 'clamp2_8': (lower_bound_7, upper_bound_7),
    # 'clamp3_8': (lower_bound_7, upper_bound_7),
    # 'clamp4_8': (lower_bound_7, upper_bound_7),
    # 'clamp5_8': (lower_bound_7, upper_bound_7),
    # 'clamp6_8': (lower_bound_7, upper_bound_7),
    # 'clamp7_8': (lower_bound_7, upper_bound_7),
    # 'clamp8_8': (lower_bound_7, upper_bound_7),
    # 'clamp9_8': (lower_bound_7, upper_bound_7),
    # 'clamp10_8': (-1, ash_bound_7),
    # 'clamp11_8': (-1, ash_bound_7),
    # 'clamp12_8': (-1, ash_bound_7),
    # 'clamp13_8': (-1, ash_bound_7),
    # 'clamp14_8': (-1, ash_bound_7),
    # 'clamp15_8': (-1, ash_bound_7),
    # 'clamp16_8': (-1, ash_bound_7),
    # 'clamp17_8': (-1, ash_bound_7),
    # 'clamp18_8': (-1, ash_bound_7),
    # 'clamp19_8': (-1, ash_bound_7),
    # 'clamp0_9': (lower_bound_8, upper_bound_8),
    # 'clamp1_9': (lower_bound_8, upper_bound_8),
    # 'clamp2_9': (lower_bound_8, upper_bound_8),
    # 'clamp3_9': (lower_bound_8, upper_bound_8),
    # 'clamp4_9': (lower_bound_8, upper_bound_8),
    # 'clamp5_9': (lower_bound_8, upper_bound_8),
    # 'clamp6_9': (lower_bound_8, upper_bound_8),
    # 'clamp7_9': (lower_bound_8, upper_bound_8),
    # 'clamp8_9': (lower_bound_8, upper_bound_8),
    # 'clamp9_9': (lower_bound_8, upper_bound_8),
    # 'clamp10_9': (-1, ash_bound_8),
    # 'clamp11_9': (-1, ash_bound_8),
    # 'clamp12_9': (-1, ash_bound_8),
    # 'clamp13_9': (-1, ash_bound_8),
    # 'clamp14_9': (-1, ash_bound_8),
    # 'clamp15_9': (-1, ash_bound_8),
    # 'clamp16_9': (-1, ash_bound_8),
    # 'clamp17_9': (-1, ash_bound_8),
    # 'clamp18_9': (-1, ash_bound_8),
    # 'clamp19_9': (-1, ash_bound_8),
    # 'clamp0_10': (lower_bound_9, upper_bound_9),
    # 'clamp1_10': (lower_bound_9, upper_bound_9),
    # 'clamp2_10': (lower_bound_9, upper_bound_9),
    # 'clamp3_10': (lower_bound_9, upper_bound_9),
    # 'clamp4_10': (lower_bound_9, upper_bound_9),
    # 'clamp5_10': (lower_bound_9, upper_bound_9),
    # 'clamp6_10': (lower_bound_9, upper_bound_9),
    # 'clamp7_10': (lower_bound_9, upper_bound_9),
    # 'clamp8_10': (lower_bound_9, upper_bound_9),
    # 'clamp9_10': (lower_bound_9, upper_bound_9),
    # 'clamp10_10': (-1, ash_bound_9),
    # 'clamp11_10': (-1, ash_bound_9),
    # 'clamp12_10': (-1, ash_bound_9),
    # 'clamp13_10': (-1, ash_bound_9),
    # 'clamp14_10': (-1, ash_bound_9),
    # 'clamp15_10': (-1, ash_bound_9),
    # 'clamp16_10': (-1, ash_bound_9),
    # 'clamp17_10': (-1, ash_bound_9),
    # 'clamp18_10': (-1, ash_bound_9),
    # 'clamp19_10': (-1, ash_bound_9),

    'clamp0_r': (lower_bound_0, upper_bound_0),
    'clamp1_r': (lower_bound_0, upper_bound_0),
    'clamp2_r': (lower_bound_0, upper_bound_0),
    'clamp3_r': (lower_bound_0, upper_bound_0),
    'clamp4_r': (lower_bound_0, upper_bound_0),
    'clamp5_r': (lower_bound_0, upper_bound_0),
    'clamp6_r': (lower_bound_0, upper_bound_0),
    'clamp7_r': (lower_bound_0, upper_bound_0),
    'clamp8_r': (lower_bound_0, upper_bound_0),
    'clamp9_r': (lower_bound_0, upper_bound_0),
    # 'clamp10_r': (lower_bound_0, upper_bound_0),
    # 'clamp11_r': (lower_bound_0, upper_bound_0),
    # 'clamp12_r': (lower_bound_0, upper_bound_0),
    # 'clamp13_r': (lower_bound_0, upper_bound_0),
    # 'clamp14_r': (lower_bound_0, upper_bound_0),
    # 'clamp15_r': (lower_bound_0, upper_bound_0),
    # 'clamp16_r': (lower_bound_0, upper_bound_0),
    # 'clamp17_r': (lower_bound_0, upper_bound_0),
    # 'clamp18_r': (lower_bound_0, upper_bound_0),
    # 'clamp19_r': (lower_bound_0, upper_bound_0),
    # 'clamp20_r': (lower_bound_0, upper_bound_0),
    # 'clamp21_r': (lower_bound_0, upper_bound_0),
    # 'clamp22_r': (lower_bound_0, upper_bound_0),
    # 'clamp23_r': (lower_bound_0, upper_bound_0),
    # 'clamp24_r': (lower_bound_0, upper_bound_0),
    # 'clamp25_r': (lower_bound_0, upper_bound_0),
    # 'clamp26_r': (lower_bound_0, upper_bound_0),
    # 'clamp27_r': (lower_bound_0, upper_bound_0),
    # 'clamp28_r': (lower_bound_0, upper_bound_0),
    # 'clamp29_r': (lower_bound_0, upper_bound_0),
    # 'clamp30_r': (lower_bound_0, upper_bound_0),
    # 'clamp31_r': (lower_bound_0, upper_bound_0),
    # 'clamp32_r': (lower_bound_0, upper_bound_0),
    # 'clamp33_r': (lower_bound_0, upper_bound_0),
    # 'clamp34_r': (lower_bound_0, upper_bound_0),

    'clamp0_a': (-1, ash_bound_0),
    'clamp1_a': (-1, ash_bound_0),
    'clamp2_a': (-1, ash_bound_0),
    'clamp3_a': (-1, ash_bound_0),
    'clamp4_a': (-1, ash_bound_0),
    'clamp5_a': (-1, ash_bound_0),
    'clamp6_a': (-1, ash_bound_0),
    'clamp7_a': (-1, ash_bound_0),
    'clamp8_a': (-1, ash_bound_0),
    'clamp9_a': (-1, ash_bound_0),
    # 'clamp10_a': (-1, ash_bound_0),
    # 'clamp11_a': (-1, ash_bound_0),
    # 'clamp12_a': (-1, ash_bound_0),
    # 'clamp13_a': (-1, ash_bound_0),
    # 'clamp14_a': (-1, ash_bound_0),
    # 'clamp15_a': (-1, ash_bound_0),
    # 'clamp16_a': (-1, ash_bound_0),
    # 'clamp17_a': (-1, ash_bound_0),
    # 'clamp18_a': (-1, ash_bound_0),
    # 'clamp19_a': (-1, ash_bound_0),
    # 'clamp20_a': (-1, ash_bound_0),
    # 'clamp21_a': (-1, ash_bound_0),
    # 'clamp22_a': (-1, ash_bound_0),
    # 'clamp23_a': (-1, ash_bound_0),
    # 'clamp24_a': (-1, ash_bound_0),
    # 'clamp25_a': (-1, ash_bound_0),
    # 'clamp26_a': (-1, ash_bound_0),
    # 'clamp27_a': (-1, ash_bound_0),
    # 'clamp28_a': (-1, ash_bound_0),
    # 'clamp29_a': (-1, ash_bound_0),
    # 'clamp30_a': (-1, ash_bound_0),
    # 'clamp31_a': (-1, ash_bound_0),
    # 'clamp32_a': (-1, ash_bound_0),
    # 'clamp33_a': (-1, ash_bound_0),
    # 'clamp34_a': (-1, ash_bound_0),
    },
    allow_duplicate_points=True,
)

acquisition_function = UtilityFunction(kind="ucb", kappa=2.576)
ood_bayesian.maximize(
    init_points=50,
    n_iter=5000,
    acquisition_function=acquisition_function,
)
