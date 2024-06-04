import numpy as np
import torch, copy, argparse, sys
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable

import models.resnet_train as resnet
from models.densenet_dice import DenseNet3
from models.wideresidual import wideresnet
from models.mobilenet import mobilenet

from utils.display_results import get_measures
from utils.svhn_loader import SVHN
from utils.tinyimages_80mn_loader import TinyImages
from models.wrn import WideResNet
from models.route import *
import torchvision.models as models
import pdb

parser = argparse.ArgumentParser(description='FFF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--batch_size', '-b', type=int, default=200, help='Batch size.')
parser.add_argument("--model", type=str, choices=["resnet50", "densenet_dice", "wideresnet", "mobilenet", "wrn", "new_resnet50", "densenet161", "wide_resnet50_2", "mobilenet_v2"])

args = parser.parse_args()

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)

if 'cifar' in args.dataset:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
else:
    mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
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

class Densenet161(nn.Module):
    def __init__(self):
        super(Densenet161, self).__init__()
        self.net = models.densenet161(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        # self.fc = nn.Sequential(*list(self.net.children())[-1:])
        self.fc = RouteDICE(2208, 1000, p=90, info="./cache/imagenet_densenet161_feat_stat.npy")

    def forward(self, x):
        out = self.extractor(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

class Wide_resnet50_2(nn.Module):
    def __init__(self):
        super(Wide_resnet50_2, self).__init__()
        self.net = models.wide_resnet50_2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        # self.fc = nn.Sequential(*list(self.net.children())[-1:])
        self.fc = RouteDICE(2048, 1000, p=90, info="./cache/imagenet_wide_resnet50_2_feat_stat.npy")
        self.emb_extractor = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        out = self.extractor(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.emb_extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        # self.fc = nn.Sequential(*list(self.net.children())[-1:])
        self.fc = RouteDICE(2048, 1000, p=90, info="./cache/imagenet_resnet50_feat_stat.npy")
        self.emb_extractor = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        out = self.extractor(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

    def pred_emb(self, x):
        out = self.emb_extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

class MobileNet_V2(nn.Module):
    def __init__(self):
        super(MobileNet_V2, self).__init__()
        self.net = models.mobilenet_v2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1], *list(self.net.children())[-1][:-1])
        # self.fc = nn.Sequential(*list(self.net.children())[-1][-1:])
        self.fc = RouteDICE(1280, 1000, p=90, info="./cache/imagenet_mobilenet_v2_feat_stat.npy")

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

    def pred_emb(self, x):
        out = self.extractor(x)
        ten = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = ten.view(ten.size(0), -1)
        return self.fc(out), out, ten

if 'cifar' in args.dataset:
    if args.dataset  == "cifar10":
        id_data = dset.CIFAR10("../data/cifar10", train=True, transform=train_transform, download=False)
        id_data_test = dset.CIFAR10("../data/cifar10", train=False, transform=id_transform, download=False)
        if args.model == "resnet50":
            model = resnet.resnet50(num_classes=10)
            model.load_state_dict(torch.load("./ckpt/resnet50_cifar10-192-best-0.9546999931335449.pth"))
        elif args.model == "densenet_dice":
            model = DenseNet3(100, 10, p=90, info="./cache/cifar10_densenet_dice_feat_stat.npy")
            model.load_state_dict(torch.load("./ckpt/checkpoint_10.pth.tar")["state_dict"])
        elif args.model == "wideresnet":
            model = wideresnet(num_classes=10)
            model.load_state_dict(torch.load("./ckpt/wideresnet_cifar10_epoch195_acc0.960599958896637.pt", map_location='cuda:0'))
        elif args.model == "mobilenet":
            model = mobilenet(class_num=10)
            model.load_state_dict(torch.load("./ckpt/mobilenet_cifar10_epoch183_acc0.90829998254776.pt", map_location='cuda:0'))
        elif args.model == "wrn":
            model = WideResNet(40, 10, 2, dropRate=0.3, p=90, info="./cache/cifar10_wrn_feat_stat.npy")
            # model.load_state_dict(torch.load("./ckpt/cifar10_wrn_pretrained_epoch_99.pt"))
            model.load_state_dict(torch.load("./ckpt/wrn_cifar10_190_best_0.9469999670982361.pth"))
        elif args.model == "new_resnet50":
            model = resnet.resnet50(num_classes=10, p=90, info="./cache/cifar10_new_resnet50_feat_stat.npy")
            model.load_state_dict(torch.load("./ckpt/new_resnet50_cifar10_180_0.9540999531745911.pth"))
        num_classes   = 10
    elif args.dataset == "cifar100":
        id_data = dset.CIFAR100("../data/cifar100", train=True, transform=train_transform, download=False)
        id_data_test = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)
        if args.model == "resnet50":
            model = resnet.resnet50(num_classes=100)
            model.load_state_dict(torch.load("./ckpt/resnet50_cifar100-196-best-0.7870000004768372.pth"))
        elif args.model == "densenet_dice":
            model = DenseNet3(100, 100, p=90, info="./cache/cifar100_densenet_dice_feat_stat.npy")
            model.load_state_dict(torch.load("./ckpt/checkpoint_100.pth.tar")["state_dict"])
        elif args.model == "wideresnet":
            model = wideresnet(num_classes=100)
            model.load_state_dict(torch.load("./ckpt/wideresnet_epoch182_acc0.7928999662399292.pt", map_location='cuda:0'))
        elif args.model == "mobilenet":
            model = mobilenet(class_num=100)
            model.load_state_dict(torch.load("./ckpt/mobilenet_epoch124_acc0.677299976348877.pt", map_location='cuda:0'))
        elif args.model == "wrn":
            model = WideResNet(40, 100, 2, dropRate=0.3, p=90, info="./cache/cifar100_wrn_feat_stat.npy")
            # model.load_state_dict(torch.load("./ckpt/cifar100_wrn_pretrained_epoch_99.pt"))
            model.load_state_dict(torch.load("./ckpt/wrn_cifar100_190_best_0.7486000061035156.pth"))
        elif args.model == "new_resnet50":
            model = resnet.resnet50(num_classes=100, p=90, info="./cache/cifar100_new_resnet50_feat_stat.npy")
            model.load_state_dict(torch.load("./ckpt/new_resnet50_cifar100_182_0.7894999980926514.pth"))
        num_classes = 100
    model = model.cuda()

    # texture_data = dset.ImageFolder(root="../data/dtd/images", transform=eval_transform)
    # places365_data = dset.ImageFolder(root="../data/places365", transform=eval_transform)
    # lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=eval_transform)
    # lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=eval_transform)
    # isun_data = dset.ImageFolder(root="../data/iSUN",transform=eval_transform)
    # svhn_data = SVHN(root="../data/svhn",transform=eval_transform, split="test", download=False)

    texture_data = dset.ImageFolder(root="../data/LSUN_pil", transform=eval_transform)
    places365_data = dset.ImageFolder(root="../data/Imagenet_pil", transform=eval_transform)
    lsunc_data = dset.ImageFolder(root="../data/Imagenet_resize", transform=eval_transform)
    lsunr_data = dset.ImageFolder(root="../data/Oxford_Pets", transform=eval_transform)
    # isun_data = dset.ImageFolder(root="../data/Stanford_Dogs",transform=eval_transform)
    isun_data = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform, download=False)
    svhn_data = dset.ImageFolder(root="../data/102flowers",transform=eval_transform)

    id_loader = torch.utils.data.DataLoader(id_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    id_loader_test = torch.utils.data.DataLoader(id_data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
    texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    id_loader  = id_loader_test
    ood_loader_list = [texture_loader, places365_loader, lsunc_loader, lsunr_loader, isun_loader, svhn_loader]
    # ood_name_list   = ['texture  ', 'places365', 'lsunc', 'lsunr', 'isun', 'svhn']
    ood_name_list   = ['LSUN_pil', 'Imagenet_pil', 'Imagenet_resize', 'Oxford_Pets', 'cifar100', '102flowers']

elif 'imagenet' in args.dataset:
    if args.model == "resnet50":
        model = ResNet50()
    elif args.model == "densenet161":
        model = Densenet161()
    elif args.model == "wide_resnet50_2":
        model = Wide_resnet50_2()
    elif args.model == "mobilenet_v2":
        model = MobileNet_V2()
    model = model.cuda()

    train_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
    num_classes      = 1000

    in_train_data    = dset.ImageFolder(root="../data/1_of_10_train", transform = train_transform)
    in_test_data     = dset.ImageFolder(root="../data/val", transform = test_transform)
    texture_data     = dset.ImageFolder(root="../data/dtd/images", transform = test_transform)
    places365_data   = dset.ImageFolder(root="../data/Places", transform = test_transform)
    inatural_data    = dset.ImageFolder(root="../data/iNaturalist", transform = test_transform) 
    sun_data         = dset.ImageFolder(root="../data/SUN", transform = test_transform)

    id_loader  = torch.utils.data.DataLoader(in_train_data,  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    id_loader_test   = torch.utils.data.DataLoader(in_test_data,   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    texture_loader   = torch.utils.data.DataLoader(texture_data,   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    inatural_loader  = torch.utils.data.DataLoader(inatural_data,  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    sun_loader       = torch.utils.data.DataLoader(sun_data,       batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)

    id_loader        = id_loader_test
    ood_loader_list  = [texture_loader, places365_loader, inatural_loader, sun_loader]
    ood_name_list    = ['texture', 'places365', 'inatural', 'sun']

def get_features(model, loader, id, total_id):
    model.eval()
    embed_tensor, logit_tensor, label_tensor, tenso_tensor = [], [], [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            logit, embed, tenso = model.pred_emb(data.cuda())
            embed_tensor.append(embed.cuda()); logit_tensor.append(logit.cuda()); label_tensor.append(target.cuda()); tenso_tensor.append(tenso.cuda())
            # sys.stdout.write('\r %d/%d | %d/%d' % (id, total_id, batch_idx, len(loader)))
        # sys.stdout.write('\r                  ')
    embed_tensor = torch.cat(embed_tensor, 0); logit_tensor = torch.cat(logit_tensor, 0); label_tensor = torch.cat(label_tensor, 0); tenso_tensor = torch.cat(tenso_tensor, 0)
    return logit_tensor.cpu(), embed_tensor.cpu(), tenso_tensor.cpu(), label_tensor.cpu()

with torch.no_grad():
    total_loader = 1 + len(ood_loader_list)
    features_id_train = get_features(model, id_loader, 0, total_loader)
    features_id = get_features(model, id_loader_test, 1, total_loader)
    features_ood_list = [get_features(model, _, idx + 2, total_loader) for idx, _ in enumerate(ood_loader_list)]

def get_msp_scores(features_list, model): 
    logit_tensor, embed_tensor, tenso_tensor, label_tensor = features_list
    logit_tensor = copy.deepcopy(logit_tensor.detach()).cuda()
    msp_scores   = - F.softmax(logit_tensor, dim=1).max(dim=1)[0]
    return msp_scores

def get_fes_scores(features_list, model):
    logit_tensor, embed_tensor, tenso_tensor, label_tensor = features_list
    logit_tensor = copy.deepcopy(logit_tensor.detach()).cuda()
    # fes_scores   = - logit_tensor.exp().mean(-1) # change
    fes_scores = - torch.logsumexp(logit_tensor, dim=1)
    return fes_scores

def get_features(model, loader, id, total_id):
    model.eval()
    embed_tensor, logit_tensor, label_tensor, tenso_tensor = [], [], [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            logit, embed, tenso = model.pred_emb(data.cuda())
            embed_tensor.append(embed.cuda()); logit_tensor.append(logit.cuda()); label_tensor.append(target.cuda()); tenso_tensor.append(tenso.cuda())
            # sys.stdout.write('\r %d/%d | %d/%d' % (id, total_id, batch_idx, len(loader)))
        # sys.stdout.write('\r                  ')
    embed_tensor = torch.cat(embed_tensor, 0); logit_tensor = torch.cat(logit_tensor, 0); label_tensor = torch.cat(label_tensor, 0); tenso_tensor = torch.cat(tenso_tensor, 0)
    return logit_tensor.cpu(), embed_tensor.cpu(), tenso_tensor.cpu(), label_tensor.cpu()

def evaluate_score(id_scores, ood_scores_list, ood_name_list):
    auroc_list, fpr_list = [], []
    for ood_scores, ood_name  in zip(ood_scores_list, ood_name_list):
        try: auroc, _, fpr = get_measures(ood_scores, id_scores)
        except: auroc, _, fpr = get_measures(ood_scores.cpu(), id_scores.cpu())
        auroc_list.append(auroc); fpr_list.append(fpr)
        print(ood_name + ' | fpr: %.2f  auroc: %.2f' % (fpr * 100, auroc * 100))
    auroc_tensor, fpr_tensor = torch.Tensor(auroc_list), torch.Tensor(fpr_list)
    print('[average] | fpr: %.2f  auroc: %.2f' % (fpr_tensor.mean() * 100, auroc_tensor.mean() * 100))
    return auroc_tensor, fpr_tensor

print('\nFES score (' + args.dataset + ')')
evaluate_score(get_fes_scores(features_id, model), [get_fes_scores(features_ood, model) for features_ood in features_ood_list], ood_name_list)
