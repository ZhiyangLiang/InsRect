import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset

import pdb
import argparse
import logging
import time
import numpy as np
import torch.optim as optim
from models.lenet import lenet
from torchvision import transforms
from utils.svhn_loader import SVHN
import sklearn.metrics as sk
from sklearn import metrics
import models
# import torchvision.models as models

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "cifar100"])
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--method", type=str, choices=["confgan", "boundarygan"])
parser.add_argument("--score", type=str, choices=["msp", "energy"])
parser.add_argument("--epoch", type=int)
args = parser.parse_args()

recall_level_default = 0.95
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

id_logits_test = []
ood_logits1 = []
ood_logits2 = []
ood_logits3 = []
ood_logits4 = []
ood_logits5 = []
ood_logits6 = []

torch.cuda.set_device(6)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

log = logging.getLogger("InsRect")

id_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])

# confgan or boundarygan: 20, 40 , 60, 80
# cnofgan + gradient_penalty: 25, 50 , 75, 100

if args.dataset == "mnist":
    id_data_test = dset.MNIST("../../data/mnist", train=False, transform=id_transform, download=False)
    # model = lenet()
    model = models.Discriminator(1, 3, 64)
    if args.method == "confgan": # confgan or cnofgan + gradient_penalty
        # model.load_state_dict(torch.load("./ckpt/lenet_mnist_epoch_%d.pth" % (args.epoch)))
        model.load_state_dict(torch.load("./ckpt/netD_lenet_mnist_epoch_%d.pth" % (args.epoch)))
    elif args.method == "boundarygan":
        model.load_state_dict(torch.load("./ckpt/lenet_mnist_epoch_%d_boundarygan.pth" % (args.epoch)))
    ood_data1 = dset.ImageFolder(root="../../data/notMNIST_small", transform=id_transform)
    ood_data2 = dset.FashionMNIST("../../data/fashionmnist", train=False, transform=id_transform, download=True)
    ood_data3 = dset.CIFAR10("../../data/cifar10", train=False, transform=id_transform, download=False)
    ood_data4 = dset.ImageFolder(root="../../data/tiny-imagenet-200/val", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor()]))
    ood_data5 = dset.ImageFolder(root="../../data/dtd/images", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
    ood_data6 = dset.ImageFolder(root="../../data/places365", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
elif args.dataset == "cifar10":
    id_data_test = dset.CIFAR10("../../data/cifar10", train=False, transform=id_transform, download=False)
    # model = models.resnet18(pretrained=False)
    # model.fc = nn.Linear(512, 10)
    model = models.Discriminator(1, 3, 64)
    if args.method == "confgan":
        # model.load_state_dict(torch.load("./ckpt/resnet18_cifar10_epoch_%d.pth" % (args.epoch)))
        model.load_state_dict(torch.load("./ckpt/netD_resnet18_cifar10_epoch_%d.pth" % (args.epoch)))
    elif args.method == "boundarygan":
        model.load_state_dict(torch.load("./ckpt/resnet18_cifar10_epoch_%d_boundarygan.pth" % (args.epoch)))

    ood_data1 = dset.CIFAR100("../../data/cifar100", train=False, transform=id_transform, download=False)
    ood_data2 = dset.ImageFolder(root="../../data/tiny-imagenet-200/val", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor()]))
    ood_data3 = dset.MNIST("../../data/mnist", train=False, transform=id_transform, download=False)
    ood_data4 = SVHN(root="../../data/svhn", transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]), split="test", download=False)
    ood_data5 = dset.ImageFolder(root="../../data/dtd/images", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
    ood_data6 = dset.ImageFolder(root="../../data/places365", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
elif args.dataset == "cifar100":
    id_data_test = dset.CIFAR100("../../data/cifar100", train=False, transform=id_transform, download=False)
    # model = models.resnet18(pretrained=False)
    # model.fc = nn.Linear(512, 100)
    model = models.Discriminator(1, 3, 64)
    if args.method == "confgan":
        # model.load_state_dict(torch.load("./ckpt/resnet18_cifar100_epoch_%d.pth" % (args.epoch)))
        model.load_state_dict(torch.load("./ckpt/netD_resnet18_cifar100_epoch_%d.pth" % (args.epoch)))
    elif args.method == "boundarygan":
        model.load_state_dict(torch.load("./ckpt/resnet18_cifar100_epoch_%d_boundarygan.pth" % (args.epoch)))

    ood_data1 = dset.CIFAR10("../../data/cifar10", train=False, transform=id_transform, download=False)
    ood_data2 = dset.ImageFolder(root="../../data/tiny-imagenet-200/val", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor()]))
    ood_data3 = dset.MNIST("../../data/mnist", train=False, transform=id_transform, download=False)
    ood_data4 = SVHN(root="../../data/svhn", transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]), split="test", download=False)
    ood_data5 = dset.ImageFolder(root="../../data/dtd/images", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
    ood_data6 = dset.ImageFolder(root="../../data/places365", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))

model = model.cuda()

# mnist = dset.MNIST("../../data/mnist", train=False, transform=id_transform, download=False)
# notmnist = dset.ImageFolder(root="../../data/notMNIST_small", transform=id_transform)
# fashionmnist = dset.FashionMNIST("../../data/fashionmnist", train=False, transform=id_transform, download=True)
# cifar10 = dset.CIFAR10("../../data/cifar10", train=False, transform=id_transform, download=False)
# cifar100 = dset.CIFAR100("../../data/cifar100", train=False, transform=id_transform, download=False)
# tiny_imagenet = dset.ImageFolder(root="../data/tiny-imagenet-200/val", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor()]))
# texture_data = dset.ImageFolder(root="../../data/dtd/images", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
# places365_data = dset.ImageFolder(root="../../data/places365", transform=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()]))
# lsunc_data = dset.ImageFolder(root="../../data/LSUN", transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
# lsunr_data = dset.ImageFolder(root="../../data/LSUN_resize", transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
# isun_data = dset.ImageFolder(root="../../data/iSUN", transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
# svhn_data = SVHN(root="../../data/svhn", transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]), split="test", download=False)


id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader1 = torch.utils.data.DataLoader(ood_data1, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader2 = torch.utils.data.DataLoader(ood_data2, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader3 = torch.utils.data.DataLoader(ood_data3, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader4 = torch.utils.data.DataLoader(ood_data4, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader5 = torch.utils.data.DataLoader(ood_data5, batch_size=args.batch_size, shuffle=True, num_workers=4)
ood_loader6 = torch.utils.data.DataLoader(ood_data6, batch_size=args.batch_size, shuffle=True, num_workers=4)

# def auc_and_fpr_recall(conf, label, tpr_th):
def auc_and_fpr_recall(conf, ood_indicator, tpr_th=0.95):
    # following convention in ML we treat OOD as positive
    # ood_indicator = np.zeros_like(label)
    # ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr

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

# def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
#     classes = np.unique(y_true)
#     if (pos_label is None and
#             not (np.array_equal(classes, [0, 1]) or
#                      np.array_equal(classes, [-1, 1]) or
#                      np.array_equal(classes, [0]) or
#                      np.array_equal(classes, [-1]) or
#                      np.array_equal(classes, [1]))):
#         raise ValueError("Data is not binary and pos_label is not specified")
#     elif pos_label is None:
#         pos_label = 1.

#     # make y_true a boolean vector
#     y_true = (y_true == pos_label)

#     # sort scores and corresponding truth values
#     desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
#     y_score = y_score[desc_score_indices]
#     y_true = y_true[desc_score_indices]

#     # y_score typically has many tied values. Here we extract
#     # the indices associated with the distinct values. We also
#     # concatenate a value for the end of the curve.
#     distinct_value_indices = np.where(np.diff(y_score))[0]
#     threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

#     # accumulate the true positives with decreasing threshold
#     tps = stable_cumsum(y_true)[threshold_idxs]
#     fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

#     thresholds = y_score[threshold_idxs]

#     recall = tps / tps[-1]

#     last_ind = tps.searchsorted(tps[-1])
#     sl = slice(last_ind, None, -1)      # [last_ind::-1]
#     recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

#     cutoff = np.argmin(np.abs(recall - recall_level))

#     # return fps[cutoff] / (np.sum(np.logical_not(y_true))) # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
#     return (fps[cutoff] / (np.sum(np.logical_not(y_true)))), thresholds[cutoff]

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    examples[np.isnan(examples)] = 0.0

    # auroc = sk.roc_auc_score(labels, examples)
    # aupr = sk.average_precision_score(labels, examples)
    # fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    # aupr_in = sk.average_precision_score(labels[:len(pos)], examples[:len(pos)])
    # aupr_out = sk.average_precision_score(labels[len(pos):], examples[len(pos):])

    # _, threshold = fpr_and_fdr_at_recall(labels, examples, recall_level)
    # auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(examples, labels, threshold)
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(examples, labels)
    return auroc, aupr_in, aupr_out, fpr

def print_measures(mylog, auroc, fpr, aupr_in, aupr_out):
    # print('& {:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr_in, 100*aupr_out))
    print('{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(100*fpr, 100*auroc, 100*aupr_in, 100*aupr_out))
    mylog.debug('& {:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr_in, 100*aupr_out))

def score_get_and_print_results(mylog, in_score, ood_score):
    model.eval()
    aurocs, aupr_ins, aupr_outs, fprs = [], [], [], []
    measures = get_measures(ood_score, in_score)
    aurocs.append(measures[0]); aupr_ins.append(measures[1]); aupr_outs.append(measures[2]); fprs.append(measures[3]);
    auroc = np.mean(aurocs); aupr_in = np.mean(aupr_ins); aupr_out = np.mean(aupr_outs); fpr = np.mean(fprs)
    print_measures(mylog, auroc, fpr, aupr_in, aupr_out)
    return fpr, auroc, aupr_in, aupr_out

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
            if args.dataset == "mnist":
                data = torch.mean(data, dim=1, keepdim=True)
            elif opt == 1:
                data = data.repeat(1, 3, 1, 1)
            feats.append(model(data))

def evaluate():
    ood_fpr1, ood_auroc1, ood_aupr_in1, ood_aupr_out1 = score_get_and_print_results(log, id_score_test, ood_score1)
    ood_fpr2, ood_auroc2, ood_aupr_in2, ood_aupr_out2 = score_get_and_print_results(log, id_score_test, ood_score2)
    ood_fpr3, ood_auroc3, ood_aupr_in3, ood_aupr_out3 = score_get_and_print_results(log, id_score_test, ood_score3)
    ood_fpr4, ood_auroc4, ood_aupr_in4, ood_aupr_out4 = score_get_and_print_results(log, id_score_test, ood_score4)
    ood_fpr5, ood_auroc5, ood_aupr_in5, ood_aupr_out5 = score_get_and_print_results(log, id_score_test, ood_score5)
    ood_fpr6, ood_auroc6, ood_aupr_in6, ood_aupr_out6 = score_get_and_print_results(log, id_score_test, ood_score6)

    print("avg_fpr: %.2f" % ((ood_fpr1 + ood_fpr2 + ood_fpr3 + ood_fpr4 + ood_fpr5 + ood_fpr6) / 6 * 100))
    print("avg_auroc: %.2f" % ((ood_auroc1 + ood_auroc2 + ood_auroc3 + ood_auroc4 + ood_auroc5 + ood_auroc6) / 6 * 100))
    print("avg_aupr_in: %.2f" % ((ood_aupr_in1 + ood_aupr_in2 + ood_aupr_in3 + ood_aupr_in4 + ood_aupr_in5 + ood_aupr_in6) / 6 * 100))
    print("avg_aupr_out: %.2f" % ((ood_aupr_out1 + ood_aupr_out2 + ood_aupr_out3 + ood_aupr_out4 + ood_aupr_out5 + ood_aupr_out6) / 6 * 100))
    

acc = test(id_loader_test)
print("acc: %.4f" % (acc))

extract_feats(id_logits_test, id_loader_test)
extract_feats(ood_logits1, ood_loader1)
extract_feats(ood_logits2, ood_loader2)
if args.dataset == "cifar10" or args.dataset == "cifar100":
    extract_feats(ood_logits3, ood_loader3, 1)
else:
    extract_feats(ood_logits3, ood_loader3)
extract_feats(ood_logits4, ood_loader4)
extract_feats(ood_logits5, ood_loader5)
extract_feats(ood_logits6, ood_loader6)

id_logits_test = torch.cat(id_logits_test, dim=0)
ood_logits1 = torch.cat(ood_logits1, dim=0)
ood_logits2 = torch.cat(ood_logits2, dim=0)
ood_logits3 = torch.cat(ood_logits3, dim=0)
ood_logits4 = torch.cat(ood_logits4, dim=0)
ood_logits5 = torch.cat(ood_logits5, dim=0)
ood_logits6 = torch.cat(ood_logits6, dim=0)

id_score_test = id_logits_test.cpu().detach().numpy()
ood_score1 = ood_logits1.cpu().detach().numpy()
ood_score2 = ood_logits2.cpu().detach().numpy()
ood_score3 = ood_logits3.cpu().detach().numpy()
ood_score4 = ood_logits4.cpu().detach().numpy()
ood_score5 = ood_logits5.cpu().detach().numpy()
ood_score6 = ood_logits6.cpu().detach().numpy()

# if args.score == "msp":
#     id_score_test =  np.max(torch.softmax(id_logits_test, axis=1).cpu().detach().numpy(), axis=1)
#     ood_score1 =  np.max(torch.softmax(ood_logits1, axis=1).cpu().detach().numpy(), axis=1)
#     ood_score2 =  np.max(torch.softmax(ood_logits2, axis=1).cpu().detach().numpy(), axis=1)
#     ood_score3 =  np.max(torch.softmax(ood_logits3, axis=1).cpu().detach().numpy(), axis=1)
#     ood_score4 =  np.max(torch.softmax(ood_logits4, axis=1).cpu().detach().numpy(), axis=1)
#     ood_score5 =  np.max(torch.softmax(ood_logits5, axis=1).cpu().detach().numpy(), axis=1)
#     ood_score6 =  np.max(torch.softmax(ood_logits6, axis=1).cpu().detach().numpy(), axis=1)
# elif args.score == "energy":
#     id_score_test =  torch.logsumexp(id_logits_test, axis=1).cpu().detach().numpy()
#     ood_score1 =  torch.logsumexp(ood_logits1, axis=1).cpu().detach().numpy()
#     ood_score2 =  torch.logsumexp(ood_logits2, axis=1).cpu().detach().numpy()
#     ood_score3 =  torch.logsumexp(ood_logits3, axis=1).cpu().detach().numpy()
#     ood_score4 =  torch.logsumexp(ood_logits4, axis=1).cpu().detach().numpy()
#     ood_score5 =  torch.logsumexp(ood_logits5, axis=1).cpu().detach().numpy()
#     ood_score6 =  torch.logsumexp(ood_logits6, axis=1).cpu().detach().numpy()

evaluate()
