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
parser.add_argument("--percent", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

final_fpr = 1.0
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

def scale(x, percentile): # nmf_relu_scale_new_version
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
# ood_feats = torch.cat(ood_feats, dim=0)[:100000]
# id_feats_test = torch.cat(id_feats_test, dim=0)
# ood1_feats = torch.cat(ood1_feats, dim=0)
# ood2_feats = torch.cat(ood2_feats, dim=0)
# ood3_feats = torch.cat(ood3_feats, dim=0)
# ood4_feats = torch.cat(ood4_feats, dim=0)
# ood5_feats = torch.cat(ood5_feats, dim=0)
# ood6_feats = torch.cat(ood6_feats, dim=0)
# torch.save(id_feats, "../data/id_feats_faster.pkl")
# torch.save(ood_feats, "../data/ood_feats_faster.pkl")
# torch.save(id_feats_test, "../data/id_feats_test_faster.pkl")
# torch.save(ood1_feats, "../data/ood1_feats_faster.pkl")
# torch.save(ood2_feats, "../data/ood2_feats_faster.pkl")
# torch.save(ood3_feats, "../data/ood3_feats_faster.pkl")
# torch.save(ood4_feats, "../data/ood4_feats_faster.pkl")
# torch.save(ood5_feats, "../data/ood5_feats_faster.pkl")
# torch.save(ood6_feats, "../data/ood6_feats_faster.pkl")

id_feats = torch.load("../data/id_feats_faster.pkl")
ood_feats = torch.load("../data/ood_feats_faster.pkl")
id_feats_test = torch.load("../data/id_feats_test_faster.pkl")
ood1_feats = torch.load("../data/ood1_feats_faster.pkl")
ood2_feats = torch.load("../data/ood2_feats_faster.pkl")
ood3_feats = torch.load("../data/ood3_feats_faster.pkl")
ood4_feats = torch.load("../data/ood4_feats_faster.pkl")
ood5_feats = torch.load("../data/ood5_feats_faster.pkl")
ood6_feats = torch.load("../data/ood6_feats_faster.pkl")

start_time_nmf = time.time()
nmf = NMF(n_components=args.num_component, max_iter=20000)
# nmf = PCA(n_components=args.num_component)
# nmf = FastICA(n_components=args.num_component, max_iter=20000)
nmf.fit(id_feats.cpu())
end_time_nmf = time.time()

ash_bound = np.percentile(nmf.components_, args.ash_percentile)
react_lower_bound = np.percentile(nmf.components_, args.lower_percentile)
react_upper_bound = np.percentile(nmf.components_, args.upper_percentile)

m_id_feats = torch.clone(id_feats)
m_ood_feats = torch.clone(ood_feats)
m_id_feats_test = torch.clone(id_feats_test)
m_ood1_feats = torch.clone(ood1_feats)
m_ood2_feats = torch.clone(ood2_feats)
m_ood3_feats = torch.clone(ood3_feats)
m_ood4_feats = torch.clone(ood4_feats)
m_ood5_feats = torch.clone(ood5_feats)
m_ood6_feats = torch.clone(ood6_feats)

id_error = m_id_feats - torch.Tensor(nmf.transform(m_id_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood_error = m_ood_feats - torch.Tensor(nmf.transform(m_ood_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
id_error_test = m_id_feats_test - torch.Tensor(nmf.transform(m_id_feats_test.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood1_error = m_ood1_feats - torch.Tensor(nmf.transform(m_ood1_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood2_error = m_ood2_feats - torch.Tensor(nmf.transform(m_ood2_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood3_error = m_ood3_feats - torch.Tensor(nmf.transform(m_ood3_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood4_error = m_ood4_feats - torch.Tensor(nmf.transform(m_ood4_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood5_error = m_ood5_feats - torch.Tensor(nmf.transform(m_ood5_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()
ood6_error = m_ood6_feats - torch.Tensor(nmf.transform(m_ood6_feats.cpu())).mm(torch.Tensor(nmf.components_)).cuda()

nmf_tran_id_feats = torch.Tensor(nmf.transform(m_id_feats.cpu())).cuda()
nmf_tran_ood_feats = torch.Tensor(nmf.transform(m_ood_feats.cpu())).cuda()
nmf_tran_id_feats_test = torch.Tensor(nmf.transform(m_id_feats_test.cpu())).cuda()
nmf_tran_ood1_feats = torch.Tensor(nmf.transform(m_ood1_feats.cpu())).cuda()
nmf_tran_ood2_feats = torch.Tensor(nmf.transform(m_ood2_feats.cpu())).cuda()
nmf_tran_ood3_feats = torch.Tensor(nmf.transform(m_ood3_feats.cpu())).cuda()
nmf_tran_ood4_feats = torch.Tensor(nmf.transform(m_ood4_feats.cpu())).cuda()
nmf_tran_ood5_feats = torch.Tensor(nmf.transform(m_ood5_feats.cpu())).cuda()
nmf_tran_ood6_feats = torch.Tensor(nmf.transform(m_ood6_feats.cpu())).cuda()

gpu_time_sum = 0

trans = torch.Tensor(nmf.components_).cuda()
final_react_list = []
final_ash_list = []

def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
# def eval_datasets(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49):
    gpu_time_start = time.time()
    global final_fpr, gpu_time_sum, id_error, ood_error, nmf_tran_id_feats, nmf_tran_ood_feats, final_react_list, final_ash_list
    global id_feats, id_feats_test, ood_feats, ood1_feats, ood2_feats, ood3_feats, ood4_feats, ood5_feats, ood6_feats
    
    m_id_feats = torch.clone(id_feats)
    m_ood_feats = torch.clone(ood_feats)
    m_trans = torch.clone(torch.Tensor(nmf.components_)).cuda()

    react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9]
    ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    # react_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49]
    # ash_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49]

    for (i, j) in zip(range(m_trans.shape[0]), react_list):
        m_trans[i] = react(m_trans[i], j)

    for (i, j) in zip(range(m_trans.shape[0]), ash_list):
        m_trans[i] = ash_s_thre(m_trans[i], j)

    m_id_feats = nmf_tran_id_feats.mm(m_trans) + id_error
    m_ood_feats = nmf_tran_ood_feats.mm(m_trans) + ood_error

    m_id_logits = model.fc(m_id_feats)
    m_ood_logits = model.fc(m_ood_feats)

    m_id_score =  - torch.logsumexp(m_id_logits, axis=1).cpu().detach().numpy()
    m_ood_score =  - torch.logsumexp(m_ood_logits, axis=1).cpu().detach().numpy()
    # if torch.isnan(torch.Tensor(m_id_score)).sum() != 0 or torch.isnan(torch.Tensor(m_ood_score)).sum() != 0:
    #     return -1e9
        # raise ValueError("The tensor has nan or inf elements")

    gpu_time_end = time.time()
    gpu_time_sum = gpu_time_sum + (gpu_time_end - gpu_time_start)
    fpr, auroc, aupr = score_get_and_print_results(log, m_id_score, m_ood_score)
    if fpr < final_fpr:
        final_fpr = fpr
        final_react_list = react_list
        final_ash_list = ash_list
    return - fpr

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
    # 'r10': (react_lower_bound, react_upper_bound),
    # 'r11': (react_lower_bound, react_upper_bound),
    # 'r12': (react_lower_bound, react_upper_bound),
    # 'r13': (react_lower_bound, react_upper_bound),
    # 'r14': (react_lower_bound, react_upper_bound),
    # 'r15': (react_lower_bound, react_upper_bound),
    # 'r16': (react_lower_bound, react_upper_bound),
    # 'r17': (react_lower_bound, react_upper_bound),
    # 'r18': (react_lower_bound, react_upper_bound),
    # 'r19': (react_lower_bound, react_upper_bound),
    # 'r20': (react_lower_bound, react_upper_bound),
    # 'r21': (react_lower_bound, react_upper_bound),
    # 'r22': (react_lower_bound, react_upper_bound),
    # 'r23': (react_lower_bound, react_upper_bound),
    # 'r24': (react_lower_bound, react_upper_bound),
    # 'r25': (react_lower_bound, react_upper_bound),
    # 'r26': (react_lower_bound, react_upper_bound),
    # 'r27': (react_lower_bound, react_upper_bound),
    # 'r28': (react_lower_bound, react_upper_bound),
    # 'r29': (react_lower_bound, react_upper_bound),
    # 'r30': (react_lower_bound, react_upper_bound),
    # 'r31': (react_lower_bound, react_upper_bound),
    # 'r32': (react_lower_bound, react_upper_bound),
    # 'r33': (react_lower_bound, react_upper_bound),
    # 'r34': (react_lower_bound, react_upper_bound),
    # 'r35': (react_lower_bound, react_upper_bound),
    # 'r36': (react_lower_bound, react_upper_bound),
    # 'r37': (react_lower_bound, react_upper_bound),
    # 'r38': (react_lower_bound, react_upper_bound),
    # 'r39': (react_lower_bound, react_upper_bound),
    # 'r40': (react_lower_bound, react_upper_bound),
    # 'r41': (react_lower_bound, react_upper_bound),
    # 'r42': (react_lower_bound, react_upper_bound),
    # 'r43': (react_lower_bound, react_upper_bound),
    # 'r44': (react_lower_bound, react_upper_bound),
    # 'r45': (react_lower_bound, react_upper_bound),
    # 'r46': (react_lower_bound, react_upper_bound),
    # 'r47': (react_lower_bound, react_upper_bound),
    # 'r48': (react_lower_bound, react_upper_bound),
    # 'r49': (react_lower_bound, react_upper_bound),

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
    # 'c10': (0, ash_bound),
    # 'c11': (0, ash_bound),
    # 'c12': (0, ash_bound),
    # 'c13': (0, ash_bound),
    # 'c14': (0, ash_bound),
    # 'c15': (0, ash_bound),
    # 'c16': (0, ash_bound),
    # 'c17': (0, ash_bound),
    # 'c18': (0, ash_bound),
    # 'c19': (0, ash_bound),
    # 'c20': (0, ash_bound),
    # 'c21': (0, ash_bound),
    # 'c22': (0, ash_bound),
    # 'c23': (0, ash_bound),
    # 'c24': (0, ash_bound),
    # 'c25': (0, ash_bound),
    # 'c26': (0, ash_bound),
    # 'c27': (0, ash_bound),
    # 'c28': (0, ash_bound),
    # 'c29': (0, ash_bound),
    # 'c30': (0, ash_bound),
    # 'c31': (0, ash_bound),
    # 'c32': (0, ash_bound),
    # 'c33': (0, ash_bound),
    # 'c34': (0, ash_bound),
    # 'c35': (0, ash_bound),
    # 'c36': (0, ash_bound),
    # 'c37': (0, ash_bound),
    # 'c38': (0, ash_bound),
    # 'c39': (0, ash_bound),
    # 'c40': (0, ash_bound),
    # 'c41': (0, ash_bound),
    # 'c42': (0, ash_bound),
    # 'c43': (0, ash_bound),
    # 'c44': (0, ash_bound),
    # 'c45': (0, ash_bound),
    # 'c46': (0, ash_bound),
    # 'c47': (0, ash_bound),
    # 'c48': (0, ash_bound),
    # 'c49': (0, ash_bound),
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

start_time_bo = time.time()

ood_bayesian.maximize(
    init_points=50,
    n_iter=50,
    acquisition_function=acquisition_function,
)

end_time_bo = time.time()

print('start_time_nmf: %.5f' % (start_time_nmf))
print('end_time_nmf: %.5f' % (end_time_nmf))
print('time_of_nmf: %.5f' % (end_time_nmf - start_time_nmf))

print('start_time_bo: %.5f' % (start_time_bo))
print('end_time_bo: %.5f' % (end_time_bo))
print('time_of_bo: %.5f' % (end_time_bo - start_time_bo))

print('time_of_bo_gpu: %.5f' % (gpu_time_sum))
print('time_of_bo_cpu: %.5f' % (end_time_bo - start_time_bo - gpu_time_sum))

def evaluate():
    global trans, final_react_list, final_ash_list
    global id_error_test, ood1_error, ood2_error, ood3_error, ood4_error, ood5_error, ood6_error
    global nmf_tran_id_feats_test, nmf_tran_ood1_feats, nmf_tran_ood2_feats, nmf_tran_ood3_feats, nmf_tran_ood4_feats, nmf_tran_ood5_feats, nmf_tran_ood6_feats
    for (i, j) in zip(range(trans.shape[0]), final_react_list):
        trans[i] = react(trans[i], j)

    for (i, j) in zip(range(trans.shape[0]), final_ash_list):
        trans[i] = ash_s_thre(trans[i], j)
    
    m_id_feats_test = nmf_tran_id_feats_test.mm(trans).cuda() + id_error_test
    m_ood1_feats = nmf_tran_ood1_feats.mm(trans).cuda() + ood1_error
    m_ood2_feats = nmf_tran_ood2_feats.mm(trans).cuda() + ood2_error
    m_ood3_feats = nmf_tran_ood3_feats.mm(trans).cuda() + ood3_error
    m_ood4_feats = nmf_tran_ood4_feats.mm(trans).cuda() + ood4_error
    m_ood5_feats = nmf_tran_ood5_feats.mm(trans).cuda() + ood5_error
    m_ood6_feats = nmf_tran_ood6_feats.mm(trans).cuda() + ood6_error

    m_id_feats_test = scale(m_id_feats_test, args.percent)
    m_ood1_feats = scale(m_ood1_feats, args.percent)
    m_ood2_feats = scale(m_ood2_feats, args.percent)
    m_ood3_feats = scale(m_ood3_feats, args.percent)
    m_ood4_feats = scale(m_ood4_feats, args.percent)
    m_ood5_feats = scale(m_ood5_feats, args.percent)
    m_ood6_feats = scale(m_ood6_feats, args.percent)

    m_id_logits_test = model.fc(m_id_feats_test)
    m_ood1_logits = model.fc(m_ood1_feats)
    m_ood2_logits = model.fc(m_ood2_feats)
    m_ood3_logits = model.fc(m_ood3_feats)
    m_ood4_logits = model.fc(m_ood4_feats)
    m_ood5_logits = model.fc(m_ood5_feats)
    m_ood6_logits = model.fc(m_ood6_feats)

    m_id_score_test =  - torch.logsumexp(m_id_logits_test, axis=1).cpu().detach().numpy()
    m_ood1_score =  - torch.logsumexp(m_ood1_logits, axis=1).cpu().detach().numpy()
    m_ood2_score =  - torch.logsumexp(m_ood2_logits, axis=1).cpu().detach().numpy()
    m_ood3_score =  - torch.logsumexp(m_ood3_logits, axis=1).cpu().detach().numpy()
    m_ood4_score =  - torch.logsumexp(m_ood4_logits, axis=1).cpu().detach().numpy()
    m_ood5_score =  - torch.logsumexp(m_ood5_logits, axis=1).cpu().detach().numpy()
    m_ood6_score =  - torch.logsumexp(m_ood6_logits, axis=1).cpu().detach().numpy()

    fpr1, auroc1, aupr1 = score_get_and_print_results(log, m_id_score_test, m_ood1_score)
    fpr2, auroc2, aupr2 = score_get_and_print_results(log, m_id_score_test, m_ood2_score)
    fpr3, auroc3, aupr3 = score_get_and_print_results(log, m_id_score_test, m_ood3_score)
    fpr4, auroc4, aupr4 = score_get_and_print_results(log, m_id_score_test, m_ood4_score)
    fpr5, auroc5, aupr5 = score_get_and_print_results(log, m_id_score_test, m_ood5_score)
    fpr6, auroc6, aupr6 = score_get_and_print_results(log, m_id_score_test, m_ood6_score)

    print("texture_fpr: %.4f; texture_auroc: %.4f" % (fpr1, auroc1))
    print("places365_fpr: %.4f; places365_auroc: %.4f" % (fpr2, auroc2))
    print("lsunr_fpr: %.4f; lsunr_auroc: %.4f" % (fpr3, auroc3))
    print("lsunc_fpr: %.4f; lsunc_auroc: %.4f" % (fpr4, auroc4))
    print("isun_fpr: %.4f; isun_auroc: %.4f" % (fpr5, auroc5))
    print("svhn_fpr: %.4f; svhn_auroc: %.4f" % (fpr6, auroc6))
    print("avg_fpr: %.4f; avg_auroc: %.4f" % ((fpr1 + fpr2 + fpr3 + fpr4 + fpr5 + fpr6) / 6, (auroc1 + auroc2 + auroc3 + auroc4 + auroc5 + auroc6)))

evaluate()
