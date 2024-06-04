import torch
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import cupy as cp
import models.hybrid_resnet as resnet
import torchvision.transforms as trn
import torchvision.datasets as dset

import pdb
import time
import itertools
from sklearn.decomposition import FastICA
# from my_ica_np import FastICA
# from my_ica_torch import FastICA

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
id_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
id_data = dset.CIFAR100("../data/cifar100", train=False, transform=id_transform)
id_dataloader = torch.utils.data.DataLoader(id_data, batch_size=200, shuffle=False, pin_memory=False)

model = resnet.resnet50(num_classes=100).cuda()
# ica = FastICA(n_components=64, max_iter=50000)
ica = FastICA(n_components=3, max_iter=50000)

start_time = time.time()
# id_feats = []
# with torch.no_grad():
#     for i, (data, label) in enumerate(id_dataloader):
#         data, label = data.cuda(), label.cuda()
#         id_feats.append(model.get_features_fc(data))

# id_feats = torch.cat(id_feats, dim=0)
# id_feats = id_feats[:2000]

id_feats = torch.Tensor([[2.934, 3.239, 1.754], [6.234, 4.576, 5.962], [7.234, 8.468, 9.943]])
id_copy = torch.clone(id_feats)
id_feats = id_feats.cpu().numpy()
print(id_feats)
new_id_feats = ica.fit_transform(id_feats)
print(id_feats)
# id_feats = id_copy
id_feats = id_copy.cpu().numpy()
print(id_feats)
print(new_id_feats)
pdb.set_trace()
end_time = time.time()
print("time: %.4f" % (end_time - start_time))
