import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import pdb
import pickle
import argparse
import time
import logging
import sklearn.metrics as sk
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import joblib
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import copy
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv

log = logging.getLogger("InsRect")
parser = argparse.ArgumentParser(description="InsRect", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["densenet161", "wide_resnet50_2", "resnet50", "mobilenet_v2"])
parser.add_argument("--num_component", type=int)
args = parser.parse_args()
recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

final_fpr = 1e9
final_avg_fpr = 1e9
final_auroc = -1e9
final_avg_auroc = -1e9

final_fpr1 = 1e9
final_fpr2 = 1e9
final_fpr3 = 1e9
final_fpr4 = 1e9
final_auroc1 = -1e9
final_auroc2 = -1e9
final_auroc3 = -1e9
final_auroc4 = -1e9

def react(x, threshold):
    x = torch.clip(x, max=threshold)
    return x

def ash_p(x, percentile):
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    t = x
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x

def ash_s_io(x, percentile): # (ash_s_percent)
    s1 = x.sum(dim=1)
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    t = x
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1])
    scale = s1 / s2
    x = x * torch.exp(scale[:, None])
    return x

# def ash_s(x, percentile): nmf_relu_scale_old_version
#     s1 = x.sum(dim=1)
#     n = x.shape[1]
#     k = int(n * percentile / 100)
#     t = x
#     v, i = torch.topk(t, k, dim=1)
#     x_relu = nmf_relu(x)
#     s2 = x_relu.sum(dim=1)
#     scale = s1 / s2 + 12
#     t.scatter_(dim=1, index=i, src=v * torch.exp(scale[:, None]))
#     return x

nmf_relu = nn.ReLU(inplace=True)
nmf_softmax = nn.Softmax(dim=1)

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

def ash_s_ins(x, percentile):
    # s1 = x.sum()
    n = x.shape[0]
    k = n - int(np.round(n * percentile / 100.0))
    t = x
    v, i = torch.topk(t, k, dim=0)
    t.zero_().scatter_(dim=0, index=i, src=v)
    # s2 = x.sum()
    # scale = s1 / s2
    # x *= torch.exp(scale)
    # x = x * scale
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

if args.model == "densenet161":
    model = Densenet161().cuda()
elif args.model == "wide_resnet50_2":
    model = Wide_resnet50_2().cuda()
elif args.model == "resnet50":
    model = ResNet50().cuda()
elif args.model == "mobilenet_v2":
    model = MobileNet_V2().cuda()

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
            feats.append(model.get_embedding_features(data))
            # feats.append(model(data))

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

if args.model == "densenet161":
    id_feats = torch.load("../data/id_mini_feats_densenet161_final.pkl")
    ood_feats = torch.load("../data/ood_feats_21kval_densenet161_final.pkl")
    ood_index_sca = torch.load("../data/index_sca3_all_densenet161.pkl")
    # ood_index_sca = torch.load("../data/index_sca5_all_densenet161.pkl")
    # ood_index_sca = torch.load("../data/index_sca10_all_densenet161.pkl")
    # ood_index_sca = torch.load("../data/index_sca15_all_densenet161.pkl")
    # ood_index_sca = torch.load("../data/index_sca20_all_densenet161.pkl")
    ood_feats = ood_feats[ood_index_sca[-300000:]]
    id_feats_test = torch.load("../data/id_feats_test_densenet161_final.pkl").cuda()
    ood1_feats = torch.load("../data/ood1_feats_densenet161_final.pkl").cuda()
    ood2_feats = torch.load("../data/ood2_feats_densenet161_final.pkl").cuda()
    ood3_feats = torch.load("../data/ood3_feats_densenet161_final.pkl").cuda()
    ood4_feats = torch.load("../data/ood4_feats_densenet161_final.pkl").cuda()
elif args.model == "wide_resnet50_2":
    id_feats = torch.load("../data/id_mini_feats_wide_resnet50_2_final.pkl")
    ood_feats = torch.load("../data/ood_feats_21kval_wide_resnet50_2_final.pkl")
    ood_index_sca = torch.load("../data/index_sca3_all_wide_resnet50_2.pkl")
    # ood_index_sca = torch.load("../data/index_sca5_all_wide_resnet50_2.pkl")
    # ood_index_sca = torch.load("../data/index_sca10_all_wide_resnet50_2.pkl")
    # ood_index_sca = torch.load("../data/index_sca15_all_wide_resnet50_2.pkl")
    # ood_index_sca = torch.load("../data/index_sca20_all_wide_resnet50_2.pkl")
    ood_feats = ood_feats[ood_index_sca[-300000:]]
    id_feats_test = torch.load("../data/id_feats_test_wide_resnet50_2_final.pkl").cuda()
    ood1_feats = torch.load("../data/ood1_feats_wide_resnet50_2_final.pkl").cuda()
    ood2_feats = torch.load("../data/ood2_feats_wide_resnet50_2_final.pkl").cuda()
    ood3_feats = torch.load("../data/ood3_feats_wide_resnet50_2_final.pkl").cuda()
    ood4_feats = torch.load("../data/ood4_feats_wide_resnet50_2_final.pkl").cuda()
elif args.model == "resnet50":
    id_feats = torch.load("../data/id_mini_feats_resnet50_final.pkl")
    ood_feats = torch.load("../data/ood_feats_21kval_resnet50_final.pkl")
    ood_index_sca = torch.load("../data/index_sca3_all_resnet50.pkl")
    # ood_index_sca = torch.load("../data/index_sca5_all_resnet50.pkl")
    # ood_index_sca = torch.load("../data/index_sca10_all_resnet50.pkl")
    # ood_index_sca = torch.load("../data/index_sca15_all_resnet50.pkl")
    # ood_index_sca = torch.load("../data/index_sca20_all_resnet50.pkl")
    ood_feats = ood_feats[ood_index_sca[-300000:]]
    id_feats_test = torch.load("../data/id_feats_test_resnet50_final.pkl").cuda()
    ood1_feats = torch.load("../data/ood1_feats_resnet50_final.pkl").cuda()
    ood2_feats = torch.load("../data/ood2_feats_resnet50_final.pkl").cuda()
    ood3_feats = torch.load("../data/ood3_feats_resnet50_final.pkl").cuda()
    ood4_feats = torch.load("../data/ood4_feats_resnet50_final.pkl").cuda()
elif args.model == "mobilenet_v2":
    id_feats = torch.load("../data/id_mini_feats_mobilenet_v2_final.pkl")
    ood_feats = torch.load("../data/ood_feats_21kval_mobilenet_v2_final.pkl")
    ood_index_sca = torch.load("../data/index_sca3_all_mobilenet_v2.pkl")
    ood_feats = ood_feats[ood_index_sca[-300000:]]
    id_feats_test = torch.load("../data/id_feats_test_mobilenet_v2_final.pkl").cuda()
    ood1_feats = torch.load("../data/ood1_feats_mobilenet_v2_final.pkl").cuda()
    ood2_feats = torch.load("../data/ood2_feats_mobilenet_v2_final.pkl").cuda()
    ood3_feats = torch.load("../data/ood3_feats_mobilenet_v2_final.pkl").cuda()
    ood4_feats = torch.load("../data/ood4_feats_mobilenet_v2_final.pkl").cuda()

id_feats = nmf_relu(id_feats)
ood_feats = nmf_relu(ood_feats)
id_feats_test = nmf_relu(id_feats_test)
ood1_feats = nmf_relu(ood1_feats)
ood2_feats = nmf_relu(ood2_feats)
ood3_feats = nmf_relu(ood3_feats)
ood4_feats = nmf_relu(ood4_feats)

if args.model == "densenet161":
    if args.num_component == 50:
        nmf = joblib.load("../data/nmf_mini_comp50_densenet161_final.pkl")
    elif args.num_component == 55:
        nmf = joblib.load("../data/nmf_mini_comp55_densenet161_final.pkl")
    elif args.num_component == 60:
        nmf = joblib.load("../data/nmf_mini_comp60_densenet161_final.pkl")
    elif args.num_component == 65:
        nmf = joblib.load("../data/nmf_mini_comp65_densenet161_final.pkl")
    elif args.num_component == 70:
        nmf = joblib.load("../data/nmf_mini_comp70_densenet161_final.pkl")
    elif args.num_component == 100:
        nmf = joblib.load("../data/nmf_mini_comp100_densenet161_final.pkl")
    elif args.num_component == 150:
        nmf = joblib.load("../data/nmf_mini_comp150_densenet161_final.pkl")
elif args.model == "wide_resnet50_2":
    if args.num_component == 50:
        nmf = joblib.load("../data/nmf_mini_comp50_wide_resnet50_2_final.pkl")
    elif args.num_component == 55:
        nmf = joblib.load("../data/nmf_mini_comp55_wide_resnet50_2_final.pkl")
    elif args.num_component == 60:
        nmf = joblib.load("../data/nmf_mini_comp60_wide_resnet50_2_final.pkl")
    elif args.num_component == 65:
        nmf = joblib.load("../data/nmf_mini_comp65_wide_resnet50_2_final.pkl")
    elif args.num_component == 70:
        nmf = joblib.load("../data/nmf_mini_comp70_wide_resnet50_2_final.pkl")
elif args.model == "resnet50":
    if args.num_component == 15:
        nmf = joblib.load("../data/nmf_mini_comp15_resnet50_final.pkl")
    elif args.num_component == 20:
        nmf = joblib.load("../data/nmf_mini_comp20_resnet50_final.pkl")
    elif args.num_component == 25:
        nmf = joblib.load("../data/nmf_mini_comp25_resnet50_final.pkl")
    elif args.num_component == 35:
        nmf = joblib.load("../data/nmf_mini_comp35_resnet50_final.pkl")
    elif args.num_component == 50:
        nmf = joblib.load("../data/nmf_mini_comp50_resnet50_final.pkl")
    elif args.num_component == 55:
        nmf = joblib.load("../data/nmf_mini_comp55_resnet50_final.pkl")
    elif args.num_component == 60:
        nmf = joblib.load("../data/nmf_mini_comp60_resnet50_final.pkl")
    elif args.num_component == 65:
        nmf = joblib.load("../data/nmf_mini_comp65_resnet50_final.pkl")
    elif args.num_component == 70:
        nmf = joblib.load("../data/nmf_mini_comp70_resnet50_final.pkl")
    elif args.num_component == 100:
        nmf = joblib.load("../data/nmf_mini_comp100_resnet50_final.pkl")
    elif args.num_component == 150:
        nmf = joblib.load("../data/nmf_mini_comp150_resnet50_final.pkl")
elif args.model == "mobilenet_v2":
    if args.num_component == 50:
        nmf = joblib.load("../data/nmf_mini_comp50_mobilenet_v2_final.pkl")

nmf_components = torch.Tensor(nmf.components_).cuda()
nmf_id_feats = torch.Tensor(nmf.transform(id_feats.cpu())).cuda()
nmf_ood_feats = torch.Tensor(nmf.transform(ood_feats.cpu())).cuda()
nmf_id_feats_test = torch.Tensor(nmf.transform(id_feats_test.cpu())).cuda()
nmf_ood1_feats = torch.Tensor(nmf.transform(ood1_feats.cpu())).cuda()
nmf_ood2_feats = torch.Tensor(nmf.transform(ood2_feats.cpu())).cuda()
nmf_ood3_feats = torch.Tensor(nmf.transform(ood3_feats.cpu())).cuda()
nmf_ood4_feats = torch.Tensor(nmf.transform(ood4_feats.cpu())).cuda()

def vim_utils(embed_tensor, logit_tensor, model):
    from sklearn.covariance import EmpiricalCovariance
    logit_tensor, embed_tensor = copy.deepcopy(logit_tensor.detach()).cuda(), copy.deepcopy(embed_tensor.detach()).cuda()
    if embed_tensor.shape[-1] >= 2048:
        DIM = 1000
    else:
        DIM = int(embed_tensor.shape[-1] / 2)
    u = -torch.matmul(torch.pinverse(model.fc[0].weight), model.fc[0].bias).detach()
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit((embed_tensor - u).cpu())
    eig_vals, eigen_vectors = torch.linalg.eig(torch.Tensor(ec.covariance_).cuda())
    eig_vals, eigen_vectors = eig_vals.float(), eigen_vectors.float()
    NS = eigen_vectors[:, torch.argsort(- eig_vals)[DIM:]]
    NS = NS.contiguous()
    vim_logit = torch.norm(torch.matmul(embed_tensor - u, NS), dim=-1)
    alpha = logit_tensor.max(dim=-1)[0].mean() / vim_logit.mean()
    return NS, alpha, u

def get_vim_scores(embed_tensor, logit_tensor, NS, alpha, u): 
    logit_tensor, embed_tensor = copy.deepcopy(logit_tensor.detach()).cuda(), copy.deepcopy(embed_tensor.detach()).cuda()
    energy = torch.logsumexp(logit_tensor, axis=-1)
    vim_logit = torch.norm(torch.matmul(embed_tensor - u, NS), dim=-1) * alpha
    vim_scores = vim_logit - energy
    return vim_scores

def residual_utils(feature_id_train, feature_id_val, model):
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    u = -torch.matmul(torch.pinverse(model.fc[0].weight), model.fc[0].bias).detach()
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit((feature_id_train - u).cpu())
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    return NS, u

def get_residual_scores(embed_tensor):
    scores = -norm(np.matmul((embed_tensor - u).cpu(), NS), axis=-1)
    return scores

with torch.no_grad():
    # id_logits = model.fc(id_feats)
    # ood_logits = model.fc(ood_feats)
    # id_logits_test = model.fc(id_feats_test)
    # ood1_logits = model.fc(ood1_feats)
    # ood2_logits = model.fc(ood2_feats)
    # ood3_logits = model.fc(ood3_feats)
    # ood4_logits = model.fc(ood4_feats)

    # id_feats = id_feats - nmf_id_feats.mm(nmf_components).cuda()
    # ood_feats = ood_feats - nmf_ood_feats.mm(nmf_components).cuda()
    id_feats_test = id_feats_test - nmf_id_feats_test.mm(nmf_components).cuda()
    ood1_feats = ood1_feats - nmf_ood1_feats.mm(nmf_components).cuda()
    ood2_feats = ood2_feats - nmf_ood2_feats.mm(nmf_components).cuda()
    ood3_feats = ood3_feats - nmf_ood3_feats.mm(nmf_components).cuda()
    ood4_feats = ood4_feats - nmf_ood4_feats.mm(nmf_components).cuda()

    # ViM    
    # NS, alpha, u = vim_utils(id_feats, id_logits, model)
    # id_scores_test = get_vim_scores(id_feats_test, id_logits_test, NS, alpha, u).cpu()
    # ood1_scores = get_vim_scores(ood1_feats, ood1_logits, NS, alpha, u).cpu()
    # ood2_scores = get_vim_scores(ood2_feats, ood2_logits, NS, alpha, u).cpu()
    # ood3_scores = get_vim_scores(ood3_feats, ood3_logits, NS, alpha, u).cpu()
    # ood4_scores = get_vim_scores(ood4_feats, ood4_logits, NS, alpha, u).cpu()
    
    # Residual
    NS, u = residual_utils(id_feats, id_feats_test, model)
    id_scores_test = get_residual_scores(id_feats_test)
    ood1_scores = get_residual_scores(ood1_feats)
    ood2_scores = get_residual_scores(ood2_feats)
    ood3_scores = get_residual_scores(ood3_feats)
    ood4_scores = get_residual_scores(ood4_feats)
    
    fpr1, auroc1, aupr1 = score_get_and_print_results(log, id_scores_test, ood1_scores)
    fpr2, auroc2, aupr2 = score_get_and_print_results(log, id_scores_test, ood2_scores)
    fpr3, auroc3, aupr3 = score_get_and_print_results(log, id_scores_test, ood3_scores)
    fpr4, auroc4, aupr4 = score_get_and_print_results(log, id_scores_test, ood4_scores)
    print("texture_fpr: %.2f; texture_auroc: %.2f" % (fpr1 * 100, auroc1 * 100))
    print("places365_fpr: %.2f; places365_auroc: %.2f" % (fpr2 * 100, auroc2 * 100))
    print("sun_fpr: %.2f; sun_auroc: %.2f" % (fpr3 * 100, auroc3 * 100))
    print("inaturalist_fpr: %.2f; inaturalist_auroc: %.2f" % (fpr4 * 100, auroc4 * 100))
    print("avg_fpr: %.2f; avg_auroc: %.2f" % ((fpr1 + fpr2 + fpr3 + fpr4) * 100 / 4, (auroc1 + auroc2 + auroc3 + auroc4) * 100 / 4))
