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
import torchvision.models as models
from utils.svhn_loader import SVHN
import numpy as np
import sklearn.metrics as sk
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from utils.tinyimages_300K_random_loader import TinyImages
from sklearn.decomposition import NMF
import math

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["resnet50", "densenet161", "wide_resnet50_2", "mobilenet_v2"])
args = parser.parse_args()

log = logging.getLogger("InsRect")
recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

id_logits_test = []
ood_logits = []
ood1_logits = []
ood2_logits = []
ood3_logits = []
ood4_logits = []

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

nmf_relu = nn.ReLU(inplace=True)
nmf_softmax = nn.Softmax(dim=1)

def ash_s(x, percentile): # nmf_relu_scale_new_version
    # x_relu = nmf_relu(x)
    s1 = x.sum(dim=1)
    # x_relu = nmf_relu(x) # after1
    n = x.shape[1]
    k = int(n * percentile / 100)
    # k = math.ceil(n * percentile / 100)
    t = x
    v, i = torch.topk(t, k, dim=1)
    s2 = v.sum(dim=1)
    x_relu = nmf_relu(x) # after2
    scale = s1 / s2
    t.scatter_(dim=1, index=i, src=v * torch.exp(scale[:, None]))
    return x

class Densenet161(nn.Module):
    def __init__(self):
        super(Densenet161, self).__init__()
        self.net = models.densenet161(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = ash_s(out, 90)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

class Wide_resnet50_2(nn.Module):
    def __init__(self):
        super(Wide_resnet50_2, self).__init__()
        self.net = models.wide_resnet50_2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = torch.flatten(out, 1)
        # out = ash_s(out, 90)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = torch.flatten(out, 1)
        # out = ash_s(out, 90)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

class MobileNet_V2(nn.Module):
    def __init__(self):
        super(MobileNet_V2, self).__init__()
        self.net = models.mobilenet_v2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1], *list(self.net.children())[-1][:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1][-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = ash_s(out, 90)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

if args.model == "resnet50":
    model = ResNet50().cuda()
elif args.model == "densenet161":
    model = Densenet161().cuda()
elif args.model == "wide_resnet50_2":
    model = Wide_resnet50_2().cuda()
elif args.model == "mobilenet_v2":
    model = MobileNet_V2().cuda()

train_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
id_data = dset.ImageFolder(root="../data/val", transform=test_transform)
# ood_data = dset.ImageFolder(root="../data/102flowers", transform=test_transform)
# ood_data = dset.ImageFolder(root="../data/imagenet21k_resized/imagenet21k_val", transform=train_transform)
# ood1_data = dset.ImageFolder(root="../data/dtd/images", transform=test_transform)
# ood2_data = dset.ImageFolder(root="../data/Places", transform=test_transform)
# ood3_data = dset.ImageFolder(root="../data/SUN", transform=test_transform)
# ood4_data = dset.ImageFolder(root="../data/iNaturalist", transform=test_transform)

id_data_loader = torch.utils.data.DataLoader(id_data, batch_size=64, shuffle=True,  num_workers=4)
# ood_data_loader = torch.utils.data.DataLoader(ood_data, batch_size=64, shuffle=True, num_workers=4)
# ood1_data_loader = torch.utils.data.DataLoader(ood1_data, batch_size=64, shuffle=True, num_workers=4)
# ood2_data_loader = torch.utils.data.DataLoader(ood2_data, batch_size=64, shuffle=True, num_workers=4)
# ood3_data_loader = torch.utils.data.DataLoader(ood3_data, batch_size=64, shuffle=True, num_workers=4)
# ood4_data_loader = torch.utils.data.DataLoader(ood4_data, batch_size=64, shuffle=True, num_workers=4)

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
            logits.append(model(data))

acc = test(id_data_loader)
print("acc: %.4f" % (acc))

# extract_logits(id_logits_test, id_data_loader)
# extract_logits(ood_logits, ood_data_loader)
# extract_logits(ood1_logits, ood1_data_loader)
# extract_logits(ood2_logits, ood2_data_loader)
# extract_logits(ood3_logits, ood3_data_loader)
# extract_logits(ood4_logits, ood4_data_loader)

# id_logits_test = torch.cat(id_logits_test, dim=0)
# ood_logits = torch.cat(ood_logits, dim=0)
# ood1_logits = torch.cat(ood1_logits, dim=0)
# ood2_logits = torch.cat(ood2_logits, dim=0)
# ood3_logits = torch.cat(ood3_logits, dim=0)
# ood4_logits = torch.cat(ood4_logits, dim=0)

# torch.save(id_logits_test, "../data/id_feats_test_densenet161_final.pkl")
# torch.save(ood_logits, "../data/ood_feats_21kval_densenet161_final.pkl")
# torch.save(ood1_logits, "../data/ood1_feats_densenet161_final.pkl")
# torch.save(ood2_logits, "../data/ood2_feats_densenet161_final.pkl")
# torch.save(ood3_logits, "../data/ood3_feats_densenet161_final.pkl")
# torch.save(ood4_logits, "../data/ood4_feats_densenet161_final.pkl")

# torch.save(id_logits_test, "../data/id_feats_test_wide_resnet50_2_final.pkl")
# torch.save(ood_logits, "../data/ood_feats_21kval_wide_resnet50_2_final.pkl")
# torch.save(ood1_logits, "../data/ood1_feats_wide_resnet50_2_final.pkl")
# torch.save(ood2_logits, "../data/ood2_feats_wide_resnet50_2_final.pkl")
# torch.save(ood3_logits, "../data/ood3_feats_wide_resnet50_2_final.pkl")
# torch.save(ood4_logits, "../data/ood4_feats_wide_resnet50_2_final.pkl")

# torch.save(id_logits_test, "../data/id_feats_test_resnet50_final.pkl")
# torch.save(ood_logits, "../data/ood_feats_21kval_resnet50_final.pkl")
# torch.save(ood1_logits, "../data/ood1_feats_resnet50_final.pkl")
# torch.save(ood2_logits, "../data/ood2_feats_resnet50_final.pkl")
# torch.save(ood3_logits, "../data/ood3_feats_resnet50_final.pkl")
# torch.save(ood4_logits, "../data/ood4_feats_resnet50_final.pkl")

# torch.save(id_logits_test, "../data/id_feats_test_mobilenet_v2_final.pkl")
# torch.save(ood_logits, "../data/ood_feats_21kval_mobilenet_v2_final.pkl")
# torch.save(ood1_logits, "../data/ood1_feats_mobilenet_v2_final.pkl")
# torch.save(ood2_logits, "../data/ood2_feats_mobilenet_v2_final.pkl")
# torch.save(ood3_logits, "../data/ood3_feats_mobilenet_v2_final.pkl")
# torch.save(ood4_logits, "../data/ood4_feats_mobilenet_v2_final.pkl")

# id_feats = torch.load("../data/id_mini_feats_densenet161_final.pkl")
# ood_feats = torch.load("../data/ood_feats_21kval_densenet161_final.pkl")
# index = torch.load("../data/index_all_densenet161.pkl")
# index = torch.load("../data/index_ash90_all_densenet161.pkl")
# ood_feats = ood_feats[index[-360000:-240000]]

# id_feats = torch.load("../data/id_mini_feats_wide_resnet50_2_final.pkl")
# ood_feats = torch.load("../data/ood_feats_21kval_wide_resnet50_2_final.pkl")
# index = torch.load("../data/index_all_wide_resnet50_2.pkl")
# index = torch.load("../data/index_ash90_all_wide_resnet50_2.pkl")
# ood_feats = ood_feats[index[-360000:-240000]]

# id_feats = torch.load("../data/id_mini_feats_resnet50_final.pkl")
# ood_feats = torch.load("../data/ood_feats_21kval_resnet50_final.pkl")
# index = torch.load("../data/index_all_resnet50.pkl")
# index = torch.load("../data/index_ash90_all_resnet50.pkl")
# index = torch.load("../data/index_oe_all_resnet50.pkl")
# ood_feats = ood_feats[index[-120000:]]

id_feats = torch.load("../data/id_mini_feats_mobilenet_v2_final.pkl")
ood_feats = torch.load("../data/ood_feats_21kval_mobilenet_v2_final.pkl")

# id_feats_test = torch.load("../data/id_feats_test_densenet161_final.pkl")
# ood1_feats = torch.load("../data/ood1_feats_densenet161_final.pkl")
# ood2_feats = torch.load("../data/ood2_feats_densenet161_final.pkl")
# ood3_feats = torch.load("../data/ood3_feats_densenet161_final.pkl")
# ood4_feats = torch.load("../data/ood4_feats_densenet161_final.pkl")

# id_feats_test = torch.load("../data/id_feats_test_wide_resnet50_2_final.pkl")
# ood1_feats = torch.load("../data/ood1_feats_wide_resnet50_2_final.pkl")
# ood2_feats = torch.load("../data/ood2_feats_wide_resnet50_2_final.pkl")
# ood3_feats = torch.load("../data/ood3_feats_wide_resnet50_2_final.pkl")
# ood4_feats = torch.load("../data/ood4_feats_wide_resnet50_2_final.pkl")

# id_feats_test = torch.load("../data/id_feats_test_resnet50_final.pkl")
# ood1_feats = torch.load("../data/ood1_feats_resnet50_final.pkl")
# ood2_feats = torch.load("../data/ood2_feats_resnet50_final.pkl")
# ood3_feats = torch.load("../data/ood3_feats_resnet50_final.pkl")
# ood4_feats = torch.load("../data/ood4_feats_resnet50_final.pkl")

# id_feats_test = nmf_softmax(id_feats_test)
# ood1_feats = nmf_softmax(ood1_feats)
# ood2_feats = nmf_softmax(ood2_feats)
# ood3_feats = nmf_softmax(ood3_feats)
# ood4_feats = nmf_softmax(ood4_feats)

id_feats = ash_s(id_feats, 3)
ood_feats = ash_s(ood_feats, 3)

# id_feats = ash_s(id_feats, 5)
# ood_feats = ash_s(ood_feats, 5)
# id_feats_test = ash_s(id_feats_test, 5)
# ood1_feats = ash_s(ood1_feats, 5)
# ood2_feats = ash_s(ood2_feats, 5)
# ood3_feats = ash_s(ood3_feats, 5)
# ood4_feats = ash_s(ood4_feats, 5)

# id_feats_test = react(id_feats_test, np.percentile(id_feats_test.cpu(), 90))
# ood1_feats = react(ood1_feats, np.percentile(id_feats_test.cpu(), 90))
# ood2_feats = react(ood2_feats, np.percentile(id_feats_test.cpu(), 90))
# ood3_feats = react(ood3_feats, np.percentile(id_feats_test.cpu(), 90))
# ood4_feats = react(ood4_feats, np.percentile(id_feats_test.cpu(), 90))

id_logits = model.fc(id_feats)
ood_logits = model.fc(ood_feats)
# id_logits_test = model.fc(id_feats_test)
# ood1_logits = model.fc(ood1_feats)
# ood2_logits = model.fc(ood2_feats)
# ood3_logits = model.fc(ood3_feats)
# ood4_logits = model.fc(ood4_feats)

# ood_scores = ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1)

id_scores = - torch.logsumexp(id_logits, axis=1).cpu().detach().numpy()
ood_scores = - torch.logsumexp(ood_logits, axis=1).cpu().detach().numpy()
ood_sorted, index = torch.sort(torch.Tensor(ood_scores))
ood_scores = ood_scores[index[-360000:-240000]]
# torch.save(index, "../data/index_all_mobilenet_v2.pkl")

if args.model == "resnet50":
    torch.save(index, "../data/index_sca3_all_resnet50.pkl")
elif args.model == "densenet161":
    torch.save(index, "../data/index_sca3_all_densenet161.pkl")
elif args.model == "wide_resnet50_2":
    torch.save(index, "../data/index_sca3_all_wide_resnet50_2.pkl")
elif args.model == "mobilenet_v2":
    torch.save(index, "../data/index_sca3_all_mobilenet_v2.pkl")

# id_scores_test = - torch.logsumexp(id_logits_test, axis=1).cpu().detach().numpy()
# ood1_scores = - torch.logsumexp(ood1_logits, axis=1).cpu().detach().numpy()
# ood2_scores = - torch.logsumexp(ood2_logits, axis=1).cpu().detach().numpy()
# ood3_scores = - torch.logsumexp(ood3_logits, axis=1).cpu().detach().numpy()
# ood4_scores = - torch.logsumexp(ood4_logits, axis=1).cpu().detach().numpy()

def evaluate():
    fpr, auroc, _ = score_get_and_print_results(log, id_scores, ood_scores)
    # ood1_fpr, ood1_auroc, _ = score_get_and_print_results(log, id_scores_test, ood1_scores)
    # ood2_fpr, ood2_auroc, _ = score_get_and_print_results(log, id_scores_test, ood2_scores)
    # ood3_fpr, ood3_auroc, _ = score_get_and_print_results(log, id_scores_test, ood3_scores)
    # ood4_fpr, ood4_auroc, _ = score_get_and_print_results(log, id_scores_test, ood4_scores)
    print("fpr: %.2f" % (fpr * 100))
    print("auroc: %.2f" % (auroc * 100))
    # print("avg_fpr: %.2f" % ((ood1_fpr + ood2_fpr + ood3_fpr + ood4_fpr) / 4 * 100))
    # print("avg_auroc: %.2f" % ((ood1_auroc + ood2_auroc + ood3_auroc + ood4_auroc) / 4 * 100))
evaluate()
