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

log = logging.getLogger("InsRect")
parser = argparse.ArgumentParser(description="InsRect", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, choices=["densenet161", "wide_resnet50_2", "resnet50", "mobilenet_v2"])
parser.add_argument("--percent", type=int)
parser.add_argument("--ash_bound", type=float)
parser.add_argument("--react_lower_bound", type=float)
parser.add_argument("--react_upper_bound", type=float)
parser.add_argument("--num_component", type=int)
parser.add_argument("--left_interval", type=int)
parser.add_argument("--right_interval", type=int)
parser.add_argument("--seed", type=int)
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
final_fpr5 = 1e9
final_fpr6 = 1e9
final_auroc1 = -1e9
final_auroc2 = -1e9
final_auroc3 = -1e9
final_auroc4 = -1e9
final_auroc5 = -1e9
final_auroc6 = -1e9

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

nmf_relu = nn.ReLU(inplace=True)

def scale(x, percentile):
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
    
    # ood_feats = torch.load("../data/ood_imagenet_o_resnet50_final.pkl")
    # ood_feats = torch.load("../data/ood_openimage_o_resnet50_final.pkl")
    
    id_feats_test = torch.load("../data/id_feats_test_resnet50_final.pkl").cuda()
    ood1_feats = torch.load("../data/ood1_feats_resnet50_final.pkl").cuda()
    ood2_feats = torch.load("../data/ood2_feats_resnet50_final.pkl").cuda()
    ood3_feats = torch.load("../data/ood3_feats_resnet50_final.pkl").cuda()
    ood4_feats = torch.load("../data/ood4_feats_resnet50_final.pkl").cuda()
    
    ood5_feats = torch.load("../data/ood_imagenet_o_resnet50_final.pkl").cuda()
    ood6_feats = torch.load("../data/ood_openimage_o_resnet50_final.pkl").cuda()
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

ood5_feats = nmf_relu(ood5_feats)
ood6_feats = nmf_relu(ood6_feats)

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

args.ash_bound = np.percentile(nmf.components_, args.ash_bound)
args.react_lower_bound = np.percentile(nmf.components_, args.react_lower_bound)
args.react_upper_bound = np.percentile(nmf.components_, args.react_upper_bound)

with torch.no_grad():
    id_error = id_feats - torch.Tensor(nmf.transform(id_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood_error = ood_feats - torch.Tensor(nmf.transform(ood_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    id_error_test = id_feats_test - torch.Tensor(nmf.transform(id_feats_test.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood1_error = ood1_feats - torch.Tensor(nmf.transform(ood1_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood2_error = ood2_feats - torch.Tensor(nmf.transform(ood2_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood3_error = ood3_feats - torch.Tensor(nmf.transform(ood3_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood4_error = ood4_feats - torch.Tensor(nmf.transform(ood4_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    
    ood5_error = ood5_feats - torch.Tensor(nmf.transform(ood5_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood6_error = ood6_feats - torch.Tensor(nmf.transform(ood6_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()

    nmf_tran_id_feats = torch.Tensor(nmf.transform(id_feats.cpu())).cuda()
    nmf_tran_ood_feats = torch.Tensor(nmf.transform(ood_feats.cpu())).cuda()
    nmf_tran_id_feats_test = torch.Tensor(nmf.transform(id_feats_test.cpu())).cuda()
    nmf_tran_ood1_feats = torch.Tensor(nmf.transform(ood1_feats.cpu())).cuda()
    nmf_tran_ood2_feats = torch.Tensor(nmf.transform(ood2_feats.cpu())).cuda()
    nmf_tran_ood3_feats = torch.Tensor(nmf.transform(ood3_feats.cpu())).cuda()
    nmf_tran_ood4_feats = torch.Tensor(nmf.transform(ood4_feats.cpu())).cuda()
    
    nmf_tran_ood5_feats = torch.Tensor(nmf.transform(ood5_feats.cpu())).cuda()
    nmf_tran_ood6_feats = torch.Tensor(nmf.transform(ood6_feats.cpu())).cuda()

# num_component = 15
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14):
# num_component = 20
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19):
# num_component = 25
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24):
# num_component = 35
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34):
# num_component = 50
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49):
# num_component = 55
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54):
# num_component = 60
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59):
# num_component = 65
def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64):
# num_component = 70
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69):
# num_component = 100
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99):
# num_component = 150
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146, r147, r148, r149, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149):
    global id_error, ood_error, id_error_test, ood1_error, ood2_error, ood3_error, ood4_error
    global nmf_tran_id_feats, nmf_tran_ood_feats, nmf_tran_id_feats_test, nmf_tran_ood1_feats, nmf_tran_ood2_feats, nmf_tran_ood3_feats, nmf_tran_ood4_feats
    global id_feats, ood_feats, id_feats_test, ood1_feats, ood2_feats, ood3_feats, ood4_feats
    global final_fpr, final_avg_fpr, final_auroc, final_avg_auroc
    global final_fpr1, final_fpr2, final_fpr3, final_fpr4
    global final_auroc1, final_auroc2, final_auroc3, final_auroc4
    global ood5_error, ood6_error, nmf_tran_ood5_feats, nmf_tran_ood6_feats, ood5_feats, ood6_feats, final_fpr5, final_fpr6, final_auroc5, final_auroc6

    m_id_feats = torch.clone(id_feats)
    m_ood_feats = torch.clone(ood_feats)
    m_trans = torch.clone(torch.Tensor(nmf.components_)).cuda()

    with torch.no_grad():
        if args.num_component == 15:
            react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14]
            ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]
        elif args.num_component == 20:
            react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19]
            ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19]
        elif args.num_component == 25:
            react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24]
            ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24]
        elif args.num_component == 35:
            react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34]
            ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34]
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
        elif args.num_component == 100:
            react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99]
            ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99]
        elif args.num_component == 150:
            react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146, r147, r148, r149]
            ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149]

        for (i, j) in zip(range(m_trans.shape[0]), react_list):
            m_trans[i] = react(m_trans[i], j)

        for (i, j) in zip(range(m_trans.shape[0]), ash_list):
            m_trans[i] = ash_s_thre(m_trans[i], j)

        m_id_feats = nmf_tran_id_feats.mm(m_trans) + id_error
        m_ood_feats = nmf_tran_ood_feats.mm(m_trans) + ood_error
    
        m_id_feats = scale(m_id_feats, args.percent)
        m_ood_feats = scale(m_ood_feats, args.percent)
    
        m_id_logits = model.fc(m_id_feats)
        m_ood_logits = model.fc(m_ood_feats)

        # m_id_scores =  - m_id_logits.max(dim=1)[0].cpu().detach().numpy()
        # m_ood_scores =  - m_ood_logits.max(dim=1)[0].cpu().detach().numpy()
        m_id_scores =  - torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
        m_ood_scores =  - torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()
        
        fpr, auroc, _ = score_get_and_print_results(log, m_id_scores, m_ood_scores)
        if fpr < final_fpr:
            final_fpr = fpr
            final_auroc = auroc
            m_id_feats_test = nmf_tran_id_feats_test.mm(m_trans) + id_error_test
            m_ood1_feats = nmf_tran_ood1_feats.mm(m_trans) + ood1_error
            m_ood2_feats = nmf_tran_ood2_feats.mm(m_trans) + ood2_error
            m_ood3_feats = nmf_tran_ood3_feats.mm(m_trans) + ood3_error
            m_ood4_feats = nmf_tran_ood4_feats.mm(m_trans) + ood4_error
        
            m_id_feats_test = scale(m_id_feats_test, args.percent)
            m_ood1_feats = scale(m_ood1_feats, args.percent)
            m_ood2_feats = scale(m_ood2_feats, args.percent)
            m_ood3_feats = scale(m_ood3_feats, args.percent)
            m_ood4_feats = scale(m_ood4_feats, args.percent)

            m_id_logits_test = model.fc(m_id_feats_test)
            m_ood1_logits = model.fc(m_ood1_feats)
            m_ood2_logits = model.fc(m_ood2_feats)
            m_ood3_logits = model.fc(m_ood3_feats)
            m_ood4_logits = model.fc(m_ood4_feats)

            m_ood5_feats = nmf_tran_ood5_feats.mm(m_trans) + ood5_error
            m_ood6_feats = nmf_tran_ood6_feats.mm(m_trans) + ood6_error
            m_ood5_feats = scale(m_ood5_feats, args.percent)
            m_ood6_feats = scale(m_ood6_feats, args.percent)
            m_ood5_logits = model.fc(m_ood5_feats)
            m_ood6_logits = model.fc(m_ood6_feats)

            # id_scores_test =  - m_id_logits_test.max(dim=1)[0].cpu().detach().numpy()
            # ood1_scores =  - m_ood1_logits.max(dim=1)[0].cpu().detach().numpy()
            # ood2_scores =  - m_ood2_logits.max(dim=1)[0].cpu().detach().numpy()
            # ood3_scores =  - m_ood3_logits.max(dim=1)[0].cpu().detach().numpy()
            # ood4_scores =  - m_ood4_logits.max(dim=1)[0].cpu().detach().numpy()
            id_scores_test =  - torch.logsumexp(m_id_logits_test, axis=1).cpu().detach().numpy()
            ood1_scores =  - torch.logsumexp(m_ood1_logits, axis=1).cpu().detach().numpy()
            ood2_scores =  - torch.logsumexp(m_ood2_logits, axis=1).cpu().detach().numpy()
            ood3_scores =  - torch.logsumexp(m_ood3_logits, axis=1).cpu().detach().numpy()
            ood4_scores =  - torch.logsumexp(m_ood4_logits, axis=1).cpu().detach().numpy()
            
            ood5_scores =  - torch.logsumexp(m_ood5_logits, axis=1).cpu().detach().numpy()
            ood6_scores =  - torch.logsumexp(m_ood6_logits, axis=1).cpu().detach().numpy()

            fpr1, auroc1, _ = score_get_and_print_results(log, id_scores_test, ood1_scores)
            fpr2, auroc2, _ = score_get_and_print_results(log, id_scores_test, ood2_scores)
            fpr3, auroc3, _ = score_get_and_print_results(log, id_scores_test, ood3_scores)
            fpr4, auroc4, _ = score_get_and_print_results(log, id_scores_test, ood4_scores)
            
            fpr5, auroc5, _ = score_get_and_print_results(log, id_scores_test, ood5_scores)
            fpr6, auroc6, _ = score_get_and_print_results(log, id_scores_test, ood6_scores)

            avg_fpr = (fpr1 + fpr2 + fpr3 + fpr4) / 4 * 100
            avg_auroc = (auroc1 + auroc2 + auroc3 + auroc4) / 4 * 100
            final_avg_fpr = avg_fpr
            final_avg_auroc = avg_auroc
            final_fpr1 = fpr1
            final_fpr2 = fpr2
            final_fpr3 = fpr3
            final_fpr4 = fpr4
            final_fpr5 = fpr5
            final_fpr6 = fpr6
            final_auroc1 = auroc1
            final_auroc2 = auroc2
            final_auroc3 = auroc3
            final_auroc4 = auroc4
            final_auroc5 = auroc5
            final_auroc6 = auroc6
            
    print("fpr: %.2f" % (final_fpr * 100))
    print("auroc: %.2f" % (final_auroc * 100))
    print("avg_fpr: %.2f" % (final_avg_fpr))
    print("avg_auroc: %.2f" % (final_avg_auroc))
    print("texture_fpr: %.2f; texture_auroc: %.2f" % (final_fpr1 * 100, final_auroc1 * 100))
    print("places365_fpr: %.2f; places365_auroc: %.2f" % (final_fpr2 * 100, final_auroc2 * 100))
    print("sun_fpr: %.2f; sun_auroc: %.2f" % (final_fpr3 * 100, final_auroc3 * 100))
    print("inaturalist_fpr: %.2f; inaturalist_auroc: %.2f" % (final_fpr4 * 100, final_auroc4 * 100))
    
    print("imagenet_o_fpr: %.2f; imagenet_o_auroc: %.2f" % (final_fpr5 * 100, final_auroc5 * 100))
    print("openimagent_o_fpr: %.2f; openimagent_o_auroc: %.2f" % (final_fpr6 * 100, final_auroc6 * 100))

    return - fpr

ood_bayesian = BayesianOptimization(
    eval_datasets,
    {
    'r0': (args.react_lower_bound, args.react_upper_bound),
    'r1': (args.react_lower_bound, args.react_upper_bound),
    'r2': (args.react_lower_bound, args.react_upper_bound),
    'r3': (args.react_lower_bound, args.react_upper_bound),
    'r4': (args.react_lower_bound, args.react_upper_bound),
    'r5': (args.react_lower_bound, args.react_upper_bound),
    'r6': (args.react_lower_bound, args.react_upper_bound),
    'r7': (args.react_lower_bound, args.react_upper_bound),
    'r8': (args.react_lower_bound, args.react_upper_bound),
    'r9': (args.react_lower_bound, args.react_upper_bound),
    'r10': (args.react_lower_bound, args.react_upper_bound),
    'r11': (args.react_lower_bound, args.react_upper_bound),
    'r12': (args.react_lower_bound, args.react_upper_bound),
    'r13': (args.react_lower_bound, args.react_upper_bound),
    'r14': (args.react_lower_bound, args.react_upper_bound),
    'r15': (args.react_lower_bound, args.react_upper_bound),
    'r16': (args.react_lower_bound, args.react_upper_bound),
    'r17': (args.react_lower_bound, args.react_upper_bound),
    'r18': (args.react_lower_bound, args.react_upper_bound),
    'r19': (args.react_lower_bound, args.react_upper_bound),
    'r20': (args.react_lower_bound, args.react_upper_bound),
    'r21': (args.react_lower_bound, args.react_upper_bound),
    'r22': (args.react_lower_bound, args.react_upper_bound),
    'r23': (args.react_lower_bound, args.react_upper_bound),
    'r24': (args.react_lower_bound, args.react_upper_bound),
    'r25': (args.react_lower_bound, args.react_upper_bound),
    'r26': (args.react_lower_bound, args.react_upper_bound),
    'r27': (args.react_lower_bound, args.react_upper_bound),
    'r28': (args.react_lower_bound, args.react_upper_bound),
    'r29': (args.react_lower_bound, args.react_upper_bound),
    'r30': (args.react_lower_bound, args.react_upper_bound),
    'r31': (args.react_lower_bound, args.react_upper_bound),
    'r32': (args.react_lower_bound, args.react_upper_bound),
    'r33': (args.react_lower_bound, args.react_upper_bound),
    'r34': (args.react_lower_bound, args.react_upper_bound),
    'r35': (args.react_lower_bound, args.react_upper_bound),
    'r36': (args.react_lower_bound, args.react_upper_bound),
    'r37': (args.react_lower_bound, args.react_upper_bound),
    'r38': (args.react_lower_bound, args.react_upper_bound),
    'r39': (args.react_lower_bound, args.react_upper_bound),
    'r40': (args.react_lower_bound, args.react_upper_bound),
    'r41': (args.react_lower_bound, args.react_upper_bound),
    'r42': (args.react_lower_bound, args.react_upper_bound),
    'r43': (args.react_lower_bound, args.react_upper_bound),
    'r44': (args.react_lower_bound, args.react_upper_bound),
    'r45': (args.react_lower_bound, args.react_upper_bound),
    'r46': (args.react_lower_bound, args.react_upper_bound),
    'r47': (args.react_lower_bound, args.react_upper_bound),
    'r48': (args.react_lower_bound, args.react_upper_bound),
    'r49': (args.react_lower_bound, args.react_upper_bound),
    'r50': (args.react_lower_bound, args.react_upper_bound),
    'r51': (args.react_lower_bound, args.react_upper_bound),
    'r52': (args.react_lower_bound, args.react_upper_bound),
    'r53': (args.react_lower_bound, args.react_upper_bound),
    'r54': (args.react_lower_bound, args.react_upper_bound),
    'r55': (args.react_lower_bound, args.react_upper_bound),
    'r56': (args.react_lower_bound, args.react_upper_bound),
    'r57': (args.react_lower_bound, args.react_upper_bound),
    'r58': (args.react_lower_bound, args.react_upper_bound),
    'r59': (args.react_lower_bound, args.react_upper_bound),
    'r60': (args.react_lower_bound, args.react_upper_bound),
    'r61': (args.react_lower_bound, args.react_upper_bound),
    'r62': (args.react_lower_bound, args.react_upper_bound),
    'r63': (args.react_lower_bound, args.react_upper_bound),
    'r64': (args.react_lower_bound, args.react_upper_bound),
    # 'r65': (args.react_lower_bound, args.react_upper_bound),
    # 'r66': (args.react_lower_bound, args.react_upper_bound),
    # 'r67': (args.react_lower_bound, args.react_upper_bound),
    # 'r68': (args.react_lower_bound, args.react_upper_bound),
    # 'r69': (args.react_lower_bound, args.react_upper_bound),
    # 'r70': (args.react_lower_bound, args.react_upper_bound),
    # 'r71': (args.react_lower_bound, args.react_upper_bound),
    # 'r72': (args.react_lower_bound, args.react_upper_bound),
    # 'r73': (args.react_lower_bound, args.react_upper_bound),
    # 'r74': (args.react_lower_bound, args.react_upper_bound),
    # 'r75': (args.react_lower_bound, args.react_upper_bound),
    # 'r76': (args.react_lower_bound, args.react_upper_bound),
    # 'r77': (args.react_lower_bound, args.react_upper_bound),
    # 'r78': (args.react_lower_bound, args.react_upper_bound),
    # 'r79': (args.react_lower_bound, args.react_upper_bound),
    # 'r80': (args.react_lower_bound, args.react_upper_bound),
    # 'r81': (args.react_lower_bound, args.react_upper_bound),
    # 'r82': (args.react_lower_bound, args.react_upper_bound),
    # 'r83': (args.react_lower_bound, args.react_upper_bound),
    # 'r84': (args.react_lower_bound, args.react_upper_bound),
    # 'r85': (args.react_lower_bound, args.react_upper_bound),
    # 'r86': (args.react_lower_bound, args.react_upper_bound),
    # 'r87': (args.react_lower_bound, args.react_upper_bound),
    # 'r88': (args.react_lower_bound, args.react_upper_bound),
    # 'r89': (args.react_lower_bound, args.react_upper_bound),
    # 'r90': (args.react_lower_bound, args.react_upper_bound),
    # 'r91': (args.react_lower_bound, args.react_upper_bound),
    # 'r92': (args.react_lower_bound, args.react_upper_bound),
    # 'r93': (args.react_lower_bound, args.react_upper_bound),
    # 'r94': (args.react_lower_bound, args.react_upper_bound),
    # 'r95': (args.react_lower_bound, args.react_upper_bound),
    # 'r96': (args.react_lower_bound, args.react_upper_bound),
    # 'r97': (args.react_lower_bound, args.react_upper_bound),
    # 'r98': (args.react_lower_bound, args.react_upper_bound),
    # 'r99': (args.react_lower_bound, args.react_upper_bound),
    # 'r100': (args.react_lower_bound, args.react_upper_bound),
    # 'r101': (args.react_lower_bound, args.react_upper_bound),
    # 'r102': (args.react_lower_bound, args.react_upper_bound),
    # 'r103': (args.react_lower_bound, args.react_upper_bound),
    # 'r104': (args.react_lower_bound, args.react_upper_bound),
    # 'r105': (args.react_lower_bound, args.react_upper_bound),
    # 'r106': (args.react_lower_bound, args.react_upper_bound),
    # 'r107': (args.react_lower_bound, args.react_upper_bound),
    # 'r108': (args.react_lower_bound, args.react_upper_bound),
    # 'r109': (args.react_lower_bound, args.react_upper_bound),
    # 'r110': (args.react_lower_bound, args.react_upper_bound),
    # 'r111': (args.react_lower_bound, args.react_upper_bound),
    # 'r112': (args.react_lower_bound, args.react_upper_bound),
    # 'r113': (args.react_lower_bound, args.react_upper_bound),
    # 'r114': (args.react_lower_bound, args.react_upper_bound),
    # 'r115': (args.react_lower_bound, args.react_upper_bound),
    # 'r116': (args.react_lower_bound, args.react_upper_bound),
    # 'r117': (args.react_lower_bound, args.react_upper_bound),
    # 'r118': (args.react_lower_bound, args.react_upper_bound),
    # 'r119': (args.react_lower_bound, args.react_upper_bound),
    # 'r120': (args.react_lower_bound, args.react_upper_bound),
    # 'r121': (args.react_lower_bound, args.react_upper_bound),
    # 'r122': (args.react_lower_bound, args.react_upper_bound),
    # 'r123': (args.react_lower_bound, args.react_upper_bound),
    # 'r124': (args.react_lower_bound, args.react_upper_bound),
    # 'r125': (args.react_lower_bound, args.react_upper_bound),
    # 'r126': (args.react_lower_bound, args.react_upper_bound),
    # 'r127': (args.react_lower_bound, args.react_upper_bound),
    # 'r128': (args.react_lower_bound, args.react_upper_bound),
    # 'r129': (args.react_lower_bound, args.react_upper_bound),
    # 'r130': (args.react_lower_bound, args.react_upper_bound),
    # 'r131': (args.react_lower_bound, args.react_upper_bound),
    # 'r132': (args.react_lower_bound, args.react_upper_bound),
    # 'r133': (args.react_lower_bound, args.react_upper_bound),
    # 'r134': (args.react_lower_bound, args.react_upper_bound),
    # 'r135': (args.react_lower_bound, args.react_upper_bound),
    # 'r136': (args.react_lower_bound, args.react_upper_bound),
    # 'r137': (args.react_lower_bound, args.react_upper_bound),
    # 'r138': (args.react_lower_bound, args.react_upper_bound),
    # 'r139': (args.react_lower_bound, args.react_upper_bound),
    # 'r140': (args.react_lower_bound, args.react_upper_bound),
    # 'r141': (args.react_lower_bound, args.react_upper_bound),
    # 'r142': (args.react_lower_bound, args.react_upper_bound),
    # 'r143': (args.react_lower_bound, args.react_upper_bound),
    # 'r144': (args.react_lower_bound, args.react_upper_bound),
    # 'r145': (args.react_lower_bound, args.react_upper_bound),
    # 'r146': (args.react_lower_bound, args.react_upper_bound),
    # 'r147': (args.react_lower_bound, args.react_upper_bound),
    # 'r148': (args.react_lower_bound, args.react_upper_bound),
    # 'r149': (args.react_lower_bound, args.react_upper_bound),

    'c0': (0, args.ash_bound),
    'c1': (0, args.ash_bound),
    'c2': (0, args.ash_bound),
    'c3': (0, args.ash_bound),
    'c4': (0, args.ash_bound),
    'c5': (0, args.ash_bound),
    'c6': (0, args.ash_bound),
    'c7': (0, args.ash_bound),
    'c8': (0, args.ash_bound),
    'c9': (0, args.ash_bound),
    'c10': (0, args.ash_bound),
    'c11': (0, args.ash_bound),
    'c12': (0, args.ash_bound),
    'c13': (0, args.ash_bound),
    'c14': (0, args.ash_bound),
    'c15': (0, args.ash_bound),
    'c16': (0, args.ash_bound),
    'c17': (0, args.ash_bound),
    'c18': (0, args.ash_bound),
    'c19': (0, args.ash_bound),
    'c20': (0, args.ash_bound),
    'c21': (0, args.ash_bound),
    'c22': (0, args.ash_bound),
    'c23': (0, args.ash_bound),
    'c24': (0, args.ash_bound),
    'c25': (0, args.ash_bound),
    'c26': (0, args.ash_bound),
    'c27': (0, args.ash_bound),
    'c28': (0, args.ash_bound),
    'c29': (0, args.ash_bound),
    'c30': (0, args.ash_bound),
    'c31': (0, args.ash_bound),
    'c32': (0, args.ash_bound),
    'c33': (0, args.ash_bound),
    'c34': (0, args.ash_bound),
    'c35': (0, args.ash_bound),
    'c36': (0, args.ash_bound),
    'c37': (0, args.ash_bound),
    'c38': (0, args.ash_bound),
    'c39': (0, args.ash_bound),
    'c40': (0, args.ash_bound),
    'c41': (0, args.ash_bound),
    'c42': (0, args.ash_bound),
    'c43': (0, args.ash_bound),
    'c44': (0, args.ash_bound),
    'c45': (0, args.ash_bound),
    'c46': (0, args.ash_bound),
    'c47': (0, args.ash_bound),
    'c48': (0, args.ash_bound),
    'c49': (0, args.ash_bound),
    'c50': (0, args.ash_bound),
    'c51': (0, args.ash_bound),
    'c52': (0, args.ash_bound),
    'c53': (0, args.ash_bound),
    'c54': (0, args.ash_bound),
    'c55': (0, args.ash_bound),
    'c56': (0, args.ash_bound),
    'c57': (0, args.ash_bound),
    'c58': (0, args.ash_bound),
    'c59': (0, args.ash_bound),
    'c60': (0, args.ash_bound),
    'c61': (0, args.ash_bound),
    'c62': (0, args.ash_bound),
    'c63': (0, args.ash_bound),
    'c64': (0, args.ash_bound),
    # 'c65': (0, args.ash_bound),
    # 'c66': (0, args.ash_bound),
    # 'c67': (0, args.ash_bound),
    # 'c68': (0, args.ash_bound),
    # 'c69': (0, args.ash_bound),
    # 'c70': (0, args.ash_bound),
    # 'c71': (0, args.ash_bound),
    # 'c72': (0, args.ash_bound),
    # 'c73': (0, args.ash_bound),
    # 'c74': (0, args.ash_bound),
    # 'c75': (0, args.ash_bound),
    # 'c76': (0, args.ash_bound),
    # 'c77': (0, args.ash_bound),
    # 'c78': (0, args.ash_bound),
    # 'c79': (0, args.ash_bound),
    # 'c80': (0, args.ash_bound),
    # 'c81': (0, args.ash_bound),
    # 'c82': (0, args.ash_bound),
    # 'c83': (0, args.ash_bound),
    # 'c84': (0, args.ash_bound),
    # 'c85': (0, args.ash_bound),
    # 'c86': (0, args.ash_bound),
    # 'c87': (0, args.ash_bound),
    # 'c88': (0, args.ash_bound),
    # 'c89': (0, args.ash_bound),
    # 'c90': (0, args.ash_bound),
    # 'c91': (0, args.ash_bound),
    # 'c92': (0, args.ash_bound),
    # 'c93': (0, args.ash_bound),
    # 'c94': (0, args.ash_bound),
    # 'c95': (0, args.ash_bound),
    # 'c96': (0, args.ash_bound),
    # 'c97': (0, args.ash_bound),
    # 'c98': (0, args.ash_bound),
    # 'c99': (0, args.ash_bound),
    # 'c100': (0, args.ash_bound),
    # 'c101': (0, args.ash_bound),
    # 'c102': (0, args.ash_bound),
    # 'c103': (0, args.ash_bound),
    # 'c104': (0, args.ash_bound),
    # 'c105': (0, args.ash_bound),
    # 'c106': (0, args.ash_bound),
    # 'c107': (0, args.ash_bound),
    # 'c108': (0, args.ash_bound),
    # 'c109': (0, args.ash_bound),
    # 'c110': (0, args.ash_bound),
    # 'c111': (0, args.ash_bound),
    # 'c112': (0, args.ash_bound),
    # 'c113': (0, args.ash_bound),
    # 'c114': (0, args.ash_bound),
    # 'c115': (0, args.ash_bound),
    # 'c116': (0, args.ash_bound),
    # 'c117': (0, args.ash_bound),
    # 'c118': (0, args.ash_bound),
    # 'c119': (0, args.ash_bound),
    # 'c120': (0, args.ash_bound),
    # 'c121': (0, args.ash_bound),
    # 'c122': (0, args.ash_bound),
    # 'c123': (0, args.ash_bound),
    # 'c124': (0, args.ash_bound),
    # 'c125': (0, args.ash_bound),
    # 'c126': (0, args.ash_bound),
    # 'c127': (0, args.ash_bound),
    # 'c128': (0, args.ash_bound),
    # 'c129': (0, args.ash_bound),
    # 'c130': (0, args.ash_bound),
    # 'c131': (0, args.ash_bound),
    # 'c132': (0, args.ash_bound),
    # 'c133': (0, args.ash_bound),
    # 'c134': (0, args.ash_bound),
    # 'c135': (0, args.ash_bound),
    # 'c136': (0, args.ash_bound),
    # 'c137': (0, args.ash_bound),
    # 'c138': (0, args.ash_bound),
    # 'c139': (0, args.ash_bound),
    # 'c140': (0, args.ash_bound),
    # 'c141': (0, args.ash_bound),
    # 'c142': (0, args.ash_bound),
    # 'c143': (0, args.ash_bound),
    # 'c144': (0, args.ash_bound),
    # 'c145': (0, args.ash_bound),
    # 'c146': (0, args.ash_bound),
    # 'c147': (0, args.ash_bound),
    # 'c148': (0, args.ash_bound),
    # 'c149': (0, args.ash_bound),
    },
    allow_duplicate_points=True,
    random_state=args.seed,
)

acquisition_function = UtilityFunction(kind="ucb", kappa=2.576)

ood_bayesian.maximize(
    init_points=50,
    n_iter=700,
    acquisition_function=acquisition_function,
)
