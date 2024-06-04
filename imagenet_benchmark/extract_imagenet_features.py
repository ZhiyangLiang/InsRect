import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

import pdb
import pickle
import argparse
import time
from sklearn.decomposition import NMF
import joblib

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, choices=["train", "test", "ood_train", "texture", "places", "sun", "inatural", "ood_val", "ood_small_classes", "mini"])
args = parser.parse_args()

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # self.net = models.resnet50(pretrained=True)
        # self.net = models.wide_resnet50_2(pretrained=True)
        # self.net = models.wide_resnet101_2(pretrained=True)
        # self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        # self.fc = nn.Sequential(*list(self.net.children())[-1:])

        self.net = models.mobilenet_v2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1], *list(self.net.children())[-1][:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1][-1:])

        # self.net = models.densenet121(pretrained=True)
        # self.net = models.densenet161(pretrained=True)
        # self.net = models.densenet169(pretrained=True)
        # self.net = models.densenet201(pretrained=True)
        # self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        # self.fc = nn.Sequential(*list(self.net.children())[-1:])

    def forward(self, x):
        out = self.extractor(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)) # mobilenet_v2 / densenet121
        out = torch.flatten(out, 1) # mobilenet_v2 / densenet121
        # out = ash_s(out, 90)
        # out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out

class ImageNetBase(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super(ImageNetBase, self).__init__(root, transform)
        self.uq_idxs = np.array(range(len(self)))
    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label

# osr_split_save_dir = './imagenet_osr_splits_winter21.pkl'

# def get_imagenet_osr_class_splits(imagenet21k_dataset, osr_split,
#                                   precomputed_split_dir=osr_split_save_dir):
#     split_to_key = {
#         'Easy': 'easy_i21k_classes',
#         'Hard': 'hard_i21k_classes'
#     }
#     # Load splits
#     with open(precomputed_split_dir, 'rb') as handle:
#         precomputed_info = pickle.load(handle)
#     osr_wnids = precomputed_info[split_to_key[osr_split]]
#     osr_wnids = set(osr_wnids)
#     selected_osr_classes_class_indices =\
#         [i for i, x in enumerate(imagenet21k_dataset.classes) if x in osr_wnids]
#     return selected_osr_classes_class_indices
# def subsample_dataset(dataset, idxs):
#     dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
#     dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
#     dataset.targets = np.array(dataset.targets)[idxs].tolist()
#     dataset.uq_idxs = dataset.uq_idxs[idxs]
#     return dataset
# def subsample_classes(dataset, include_classes=list(range(1000))):
#     cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]
#     target_xform_dict = {}
#     for i, k in enumerate(include_classes):
#         target_xform_dict[k] = i
#     dataset = subsample_dataset(dataset, cls_idxs)
#     dataset.target_transform = lambda x: target_xform_dict[x]
#     return dataset

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

train_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
if args.dataset == "train":
    data = dset.ImageFolder(root="../data/ImageNet/imagenet/train", transform=train_transform)
elif args.dataset == "test":
    data = dset.ImageFolder(root="../data/ImageNet/imagenet/val", transform=test_transform)
elif args.dataset == "mini":
    data = dset.ImageFolder(root="../data/1_of_10_train", transform=train_transform)
elif args.dataset == "ood_train":
    data = ImageNetBase(root="../data/imagenet21k_resized/imagenet21k_train", transform=train_transform)
elif args.dataset == "ood_val":
    data = ImageNetBase(root="../data/imagenet21k_resized/imagenet21k_val", transform=train_transform)
elif args.dataset == "ood_small_classes":
    data = ImageNetBase(root="../data/imagenet21k_resized/imagenet21k_small_classes", transform=train_transform)
elif args.dataset == "texture":
    data = dset.ImageFolder(root="../data/dtd/images", transform=test_transform)
elif args.dataset == "places":
    data = dset.ImageFolder(root="../data/Places", transform=test_transform)
elif args.dataset == "sun":
    data = dset.ImageFolder(root="../data/SUN", transform=test_transform)
elif args.dataset == "inatural":
    data = dset.ImageFolder(root="../data/iNaturalist", transform=test_transform)

data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True,  num_workers=4)

model = MyNet().cuda()
feats_final = []

with torch.no_grad():
    for idx, (data, target) in enumerate(data_loader):
        torch.cuda.empty_cache()
        data, target = data.cuda(), target.cuda()
        feat = model(data)
        feats_final.append(feat)
        # if idx > 40036:
        #     break

feats_final = torch.cat(feats_final, 0)
if args.dataset == "train":
    torch.save(feats_final, "../data/id_train_feats.pkl")
elif args.dataset == "test":
    torch.save(feats_final, "../data/id_test_feats.pkl")
elif args.dataset == "mini":
    # torch.save(feats_final, "../data/id_mini_feats_densenet161_final.pkl")
    # torch.save(feats_final, "../data/id_mini_feats_wide_resnet50_2_final.pkl")
    # torch.save(feats_final, "../data/id_mini_feats_resnet50_final.pkl")
    torch.save(feats_final, "../data/id_mini_feats_mobilenet_v2_final.pkl")
elif args.dataset == "ood_train":
    torch.save(feats_final, "../data/id_ood_feats.pkl")
elif args.dataset == "ood_val":
    torch.save(feats_final, "../data/ood_val_feats_densenet161_final.pkl")
elif args.dataset == "ood_small_classes":
    torch.save(feats_final, "../data/ood_small_classes_feats.pkl")
elif args.dataset == "texture":
    torch.save(feats_final, "../data/texture_feats.pkl")
elif args.dataset == "places":
    torch.save(feats_final, "../data/places_feats.pkl")
elif args.dataset == "sun":
    torch.save(feats_final, "../data/sun_feats.pkl")
elif args.dataset == "inatural":
    torch.save(feats_final, "../data/inatural_feats.pkl")

# feats_final = torch.load("../data/id_mini_feats_densenet161_final.pkl")
# feats_final = torch.load("../data/id_mini_feats_wide_resnet50_2_final.pkl")
# feats_final = torch.load("../data/id_mini_feats_resnet50_final.pkl")
# feats_final = torch.load("../data/id_mini_feats_mobilenet_v2_final.pkl")
# pdb.set_trace()

print(feats_final.shape)
nmf_relu = nn.ReLU(inplace=True)
nmf = NMF(n_components=50, max_iter=50000)
feats_final = nmf_relu(feats_final)
nmf.fit(feats_final.cpu())
print(nmf.components_)
# joblib.dump(nmf, "../data/nmf_mini_comp50_densenet161_final.pkl")
# joblib.dump(nmf, "../data/nmf_mini_comp50_wide_resnet50_2_final.pkl")
# joblib.dump(nmf, "../data/nmf_mini_comp50_resnet50_final.pkl")
joblib.dump(nmf, "../data/nmf_mini_comp50_mobilenet_v2_final.pkl")
