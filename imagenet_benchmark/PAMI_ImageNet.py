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
parser.add_argument("--num_component", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()
recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def react(x, threshold):
    x = torch.clip(x, max=threshold)
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
    ood_feats = ood_feats[ood_index_sca[-300000:]]
    
    id_feats_test = torch.load("../data/id_feats_test_resnet50_final.pkl").cuda()
    ood1_feats = torch.load("../data/ood1_feats_resnet50_final.pkl").cuda()
    ood2_feats = torch.load("../data/ood2_feats_resnet50_final.pkl").cuda()
    ood3_feats = torch.load("../data/ood3_feats_resnet50_final.pkl").cuda()
    ood4_feats = torch.load("../data/ood4_feats_resnet50_final.pkl").cuda()
    
    # ood5_feats = torch.load("../data/ood_imagenet_o_resnet50_final.pkl").cuda()
    # ood6_feats = torch.load("../data/ood_openimage_o_resnet50_final.pkl").cuda()
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

# ood5_feats = nmf_relu(ood5_feats)
# ood6_feats = nmf_relu(ood6_feats)

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

print(1)

with torch.no_grad():
    id_error = id_feats - torch.Tensor(nmf.transform(id_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood_error = ood_feats - torch.Tensor(nmf.transform(ood_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    id_error_test = id_feats_test - torch.Tensor(nmf.transform(id_feats_test.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood1_error = ood1_feats - torch.Tensor(nmf.transform(ood1_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood2_error = ood2_feats - torch.Tensor(nmf.transform(ood2_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood3_error = ood3_feats - torch.Tensor(nmf.transform(ood3_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    ood4_error = ood4_feats - torch.Tensor(nmf.transform(ood4_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()

    # ood5_error = ood5_feats - torch.Tensor(nmf.transform(ood5_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
    # ood6_error = ood6_feats - torch.Tensor(nmf.transform(ood6_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()

#     # torch.save(id_error, "../data/intuitive_imagenet_id_error")
#     # torch.save(ood_error, "../data/intuitive_imagenet_ood_error")
#     # torch.save(id_error_test, "../data/intuitive_imagenet_id_error_test")
#     # torch.save(ood1_error, "../data/intuitive_imagenet_ood1_error")
#     # torch.save(ood2_error, "../data/intuitive_imagenet_ood2_error")
#     # torch.save(ood3_error, "../data/intuitive_imagenet_ood3_error")
#     # torch.save(ood4_error, "../data/intuitive_imagenet_ood4_error")
#     # torch.save(ood5_error, "../data/intuitive_imagenet_ood5_error")
#     # torch.save(ood6_error, "../data/intuitive_imagenet_ood6_error")
#     # torch.save(id_error, "../data/intuitive_imagenet_id_error_num65")
#     # torch.save(ood_error, "../data/intuitive_imagenet_ood_error_num65")
#     # torch.save(id_error_test, "../data/intuitive_imagenet_id_error_test_num65")
#     # torch.save(ood1_error, "../data/intuitive_imagenet_ood1_error_num65")
#     # torch.save(ood2_error, "../data/intuitive_imagenet_ood2_error_num65")
#     # torch.save(ood3_error, "../data/intuitive_imagenet_ood3_error_num65")
#     # torch.save(ood4_error, "../data/intuitive_imagenet_ood4_error_num65")
#     # torch.save(ood5_error, "../data/intuitive_imagenet_ood5_error_num65")
#     # torch.save(ood6_error, "../data/intuitive_imagenet_ood6_error_num65")
#     torch.save(id_error, "../data/intuitive_imagenet_id_error_num100")
#     torch.save(ood_error, "../data/intuitive_imagenet_ood_error_num100")
#     torch.save(id_error_test, "../data/intuitive_imagenet_id_error_test_num100")
#     torch.save(ood1_error, "../data/intuitive_imagenet_ood1_error_num100")
#     torch.save(ood2_error, "../data/intuitive_imagenet_ood2_error_num100")
#     torch.save(ood3_error, "../data/intuitive_imagenet_ood3_error_num100")
#     torch.save(ood4_error, "../data/intuitive_imagenet_ood4_error_num100")
#     torch.save(ood5_error, "../data/intuitive_imagenet_ood5_error_num100")
#     torch.save(ood6_error, "../data/intuitive_imagenet_ood6_error_num100")

    nmf_tran_id_feats = torch.Tensor(nmf.transform(id_feats.cpu())).cuda()
    nmf_tran_ood_feats = torch.Tensor(nmf.transform(ood_feats.cpu())).cuda()
    nmf_tran_id_feats_test = torch.Tensor(nmf.transform(id_feats_test.cpu())).cuda()
    nmf_tran_ood1_feats = torch.Tensor(nmf.transform(ood1_feats.cpu())).cuda()
    nmf_tran_ood2_feats = torch.Tensor(nmf.transform(ood2_feats.cpu())).cuda()
    nmf_tran_ood3_feats = torch.Tensor(nmf.transform(ood3_feats.cpu())).cuda()
    nmf_tran_ood4_feats = torch.Tensor(nmf.transform(ood4_feats.cpu())).cuda()
    
#     nmf_tran_ood5_feats = torch.Tensor(nmf.transform(ood5_feats.cpu())).cuda()
#     nmf_tran_ood6_feats = torch.Tensor(nmf.transform(ood6_feats.cpu())).cuda()

#     # torch.save(nmf_tran_id_feats, "../data/intuitive_imagenet_nmf_tran_id_feats")
#     # torch.save(nmf_tran_ood_feats, "../data/intuitive_imagenet_nmf_tran_ood_feats")
#     # torch.save(nmf_tran_id_feats_test, "../data/intuitive_imagenet_nmf_tran_id_feats_test")
#     # torch.save(nmf_tran_ood1_feats, "../data/intuitive_imagenet_nmf_tran_ood1_feats")
#     # torch.save(nmf_tran_ood2_feats, "../data/intuitive_imagenet_nmf_tran_ood2_feats")
#     # torch.save(nmf_tran_ood3_feats, "../data/intuitive_imagenet_nmf_tran_ood3_feats")
#     # torch.save(nmf_tran_ood4_feats, "../data/intuitive_imagenet_nmf_tran_ood4_feats")
#     # torch.save(nmf_tran_ood5_feats, "../data/intuitive_imagenet_nmf_tran_ood5_feats")
#     # torch.save(nmf_tran_ood6_feats, "../data/intuitive_imagenet_nmf_tran_ood6_feats")
#     # torch.save(nmf_tran_id_feats, "../data/intuitive_imagenet_nmf_tran_id_feats_num65")
#     # torch.save(nmf_tran_ood_feats, "../data/intuitive_imagenet_nmf_tran_ood_feats_num65")
#     # torch.save(nmf_tran_id_feats_test, "../data/intuitive_imagenet_nmf_tran_id_feats_test_num65")
#     # torch.save(nmf_tran_ood1_feats, "../data/intuitive_imagenet_nmf_tran_ood1_feats_num65")
#     # torch.save(nmf_tran_ood2_feats, "../data/intuitive_imagenet_nmf_tran_ood2_feats_num65")
#     # torch.save(nmf_tran_ood3_feats, "../data/intuitive_imagenet_nmf_tran_ood3_feats_num65")
#     # torch.save(nmf_tran_ood4_feats, "../data/intuitive_imagenet_nmf_tran_ood4_feats_num65")
#     # torch.save(nmf_tran_ood5_feats, "../data/intuitive_imagenet_nmf_tran_ood5_feats_num65")
#     # torch.save(nmf_tran_ood6_feats, "../data/intuitive_imagenet_nmf_tran_ood6_feats_num65")
#     torch.save(nmf_tran_id_feats, "../data/intuitive_imagenet_nmf_tran_id_feats_num100")
#     torch.save(nmf_tran_ood_feats, "../data/intuitive_imagenet_nmf_tran_ood_feats_num100")
#     torch.save(nmf_tran_id_feats_test, "../data/intuitive_imagenet_nmf_tran_id_feats_test_num100")
#     torch.save(nmf_tran_ood1_feats, "../data/intuitive_imagenet_nmf_tran_ood1_feats_num100")
#     torch.save(nmf_tran_ood2_feats, "../data/intuitive_imagenet_nmf_tran_ood2_feats_num100")
#     torch.save(nmf_tran_ood3_feats, "../data/intuitive_imagenet_nmf_tran_ood3_feats_num100")
#     torch.save(nmf_tran_ood4_feats, "../data/intuitive_imagenet_nmf_tran_ood4_feats_num100")
#     torch.save(nmf_tran_ood5_feats, "../data/intuitive_imagenet_nmf_tran_ood5_feats_num100")
#     torch.save(nmf_tran_ood6_feats, "../data/intuitive_imagenet_nmf_tran_ood6_feats_num100")

# id_error = torch.load("../data/intuitive_imagenet_id_error_num65")
# ood_error = torch.load("../data/intuitive_imagenet_ood_error_num65")
# id_error_test = torch.load("../data/intuitive_imagenet_id_error_test_num65")
# ood1_error = torch.load("../data/intuitive_imagenet_ood1_error_num65")
# ood2_error = torch.load("../data/intuitive_imagenet_ood2_error_num65")
# ood3_error = torch.load("../data/intuitive_imagenet_ood3_error_num65")
# ood4_error = torch.load("../data/intuitive_imagenet_ood4_error_num65")
# ood5_error = torch.load("../data/intuitive_imagenet_ood5_error_num65")
# ood6_error = torch.load("../data/intuitive_imagenet_ood6_error_num65")
# nmf_tran_id_feats = torch.load("../data/intuitive_imagenet_nmf_tran_id_feats_num65")
# nmf_tran_ood_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood_feats_num65")
# nmf_tran_id_feats_test = torch.load("../data/intuitive_imagenet_nmf_tran_id_feats_test_num65")
# nmf_tran_ood1_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood1_feats_num65")
# nmf_tran_ood2_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood2_feats_num65")
# nmf_tran_ood3_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood3_feats_num65")
# nmf_tran_ood4_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood4_feats_num65")
# nmf_tran_ood5_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood5_feats_num65")
# nmf_tran_ood6_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood6_feats_num65")

# id_error = torch.load("../data/intuitive_imagenet_id_error_num100")
# ood_error = torch.load("../data/intuitive_imagenet_ood_error_num100")
# id_error_test = torch.load("../data/intuitive_imagenet_id_error_test_num100")
# ood1_error = torch.load("../data/intuitive_imagenet_ood1_error_num100")
# ood2_error = torch.load("../data/intuitive_imagenet_ood2_error_num100")
# ood3_error = torch.load("../data/intuitive_imagenet_ood3_error_num100")
# ood4_error = torch.load("../data/intuitive_imagenet_ood4_error_num100")
# ood5_error = torch.load("../data/intuitive_imagenet_ood5_error_num100")
# ood6_error = torch.load("../data/intuitive_imagenet_ood6_error_num100")
# nmf_tran_id_feats = torch.load("../data/intuitive_imagenet_nmf_tran_id_feats_num100")
# nmf_tran_ood_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood_feats_num100")
# nmf_tran_id_feats_test = torch.load("../data/intuitive_imagenet_nmf_tran_id_feats_test_num100")
# nmf_tran_ood1_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood1_feats_num100")
# nmf_tran_ood2_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood2_feats_num100")
# nmf_tran_ood3_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood3_feats_num100")
# nmf_tran_ood4_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood4_feats_num100")
# nmf_tran_ood5_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood5_feats_num100")
# nmf_tran_ood6_feats = torch.load("../data/intuitive_imagenet_nmf_tran_ood6_feats_num100")
trans = torch.clone(torch.Tensor(nmf.components_)).cuda()
print(2)

def evaluate():
    global trans
    global id_error, ood_error, id_error_test, ood1_error, ood2_error, ood3_error, ood4_error, ood5_error, ood6_error
    global nmf_tran_id_feats, nmf_tran_ood_feats, nmf_tran_id_feats_test, nmf_tran_ood1_feats, nmf_tran_ood2_feats, nmf_tran_ood3_feats, nmf_tran_ood4_feats, nmf_tran_ood5_feats, nmf_tran_ood6_feats
    
    m_trans = torch.clone(trans) # test
    s_trans = torch.sum(m_trans, dim=1)
    # s_trans = m_trans.norm(p=1, dim=1)
    # s_trans = m_trans.norm(p=2, dim=1)
    num_trans = (m_trans > 100).sum()
    w_id_feats = torch.sum(nmf_tran_id_feats, dim=0)
    w_ood_feats = torch.sum(nmf_tran_ood_feats, dim=0)

    for (i, j) in zip(range(m_trans.shape[0]), (((w_id_feats - w_ood_feats) * s_trans) < np.percentile(((w_id_feats - w_ood_feats) * s_trans).cpu(), 10))):
        if j:
            m_trans[i] = react(m_trans[i], 0)
    
    print(3)
    m_id_feats = nmf_tran_id_feats.mm(m_trans).cuda() + id_error
    m_ood_feats = nmf_tran_ood_feats.mm(m_trans).cuda() + ood_error
    m_id_feats_test = nmf_tran_id_feats_test.mm(m_trans).cuda() + id_error_test
    m_ood1_feats = nmf_tran_ood1_feats.mm(m_trans).cuda() + ood1_error
    m_ood2_feats = nmf_tran_ood2_feats.mm(m_trans).cuda() + ood2_error
    m_ood3_feats = nmf_tran_ood3_feats.mm(m_trans).cuda() + ood3_error
    m_ood4_feats = nmf_tran_ood4_feats.mm(m_trans).cuda() + ood4_error
    # m_ood5_feats = nmf_tran_ood5_feats.mm(m_trans).cuda() + ood5_error
    # m_ood6_feats = nmf_tran_ood6_feats.mm(m_trans).cuda() + ood6_error

    print(4)
    m_id_feats_test = scale(m_id_feats_test, args.percent)
    m_ood1_feats = scale(m_ood1_feats, args.percent)
    m_ood2_feats = scale(m_ood2_feats, args.percent)
    m_ood3_feats = scale(m_ood3_feats, args.percent)
    m_ood4_feats = scale(m_ood4_feats, args.percent)
    # m_ood5_feats = scale(m_ood5_feats, args.percent)
    # m_ood6_feats = scale(m_ood6_feats, args.percent)

    print(5)
    m_id_logits = model.fc(m_id_feats)
    m_ood_logits = model.fc(m_ood_feats)
    m_id_score = -torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
    m_ood_score = -torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()

    m_id_logits_test = model.fc(m_id_feats_test)
    m_ood1_logits = model.fc(m_ood1_feats)
    m_ood2_logits = model.fc(m_ood2_feats)
    m_ood3_logits = model.fc(m_ood3_feats)
    m_ood4_logits = model.fc(m_ood4_feats)
    # m_ood5_logits = model.fc(m_ood5_feats)
    # m_ood6_logits = model.fc(m_ood6_feats)
    m_id_score_test =  - torch.logsumexp(m_id_logits_test, axis=1).cpu().detach().numpy()
    m_ood1_score =  - torch.logsumexp(m_ood1_logits, axis=1).cpu().detach().numpy()
    m_ood2_score =  - torch.logsumexp(m_ood2_logits, axis=1).cpu().detach().numpy()
    m_ood3_score =  - torch.logsumexp(m_ood3_logits, axis=1).cpu().detach().numpy()
    m_ood4_score =  - torch.logsumexp(m_ood4_logits, axis=1).cpu().detach().numpy()
    # m_ood5_score =  - torch.logsumexp(m_ood5_logits, axis=1).cpu().detach().numpy()
    # m_ood6_score =  - torch.logsumexp(m_ood6_logits, axis=1).cpu().detach().numpy()

    fpr, auroc, aupr = score_get_and_print_results(log, m_id_score, m_ood_score)
    fpr1, auroc1, aupr1 = score_get_and_print_results(log, m_id_score_test, m_ood1_score)
    fpr2, auroc2, aupr2 = score_get_and_print_results(log, m_id_score_test, m_ood2_score)
    fpr3, auroc3, aupr3 = score_get_and_print_results(log, m_id_score_test, m_ood3_score)
    fpr4, auroc4, aupr4 = score_get_and_print_results(log, m_id_score_test, m_ood4_score)
    # fpr5, auroc5, aupr5 = score_get_and_print_results(log, m_id_score_test, m_ood5_score)
    # fpr6, auroc6, aupr6 = score_get_and_print_results(log, m_id_score_test, m_ood6_score)

    print("texture_fpr: %.4f; texture_auroc: %.4f" % (fpr1 * 100, auroc1 * 100))
    print("places365_fpr: %.4f; places365_auroc: %.4f" % (fpr2 * 100, auroc2 * 100))
    print("sun_fpr: %.4f; sun_auroc: %.4f" % (fpr3 * 100, auroc3 * 100))
    print("inaturalist_fpr: %.4f; inaturalist_auroc: %.4f" % (fpr4 * 100, auroc4 * 100))
    
    # print("imagenet_o_fpr: %.4f; imagenet_o_auroc: %.4f" % (fpr5 * 100, auroc5 * 100))
    # print("openimagent_o_fpr: %.4f; openimagent_o_auroc: %.4f" % (fpr6 * 100, auroc6 * 100))
    print("avg_fpr: %.4f; avg_auroc: %.4f" % ((fpr1 + fpr2 + fpr3 + fpr4) / 4, (auroc1 + auroc2 + auroc3 + auroc4) / 4))

evaluate()
