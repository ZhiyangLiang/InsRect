import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torchvision.transforms as trn
import torchvision.datasets as dset

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])
        self.fc = nn.Sequential(*list(self.net.children())[-1:])
        
    def forward(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_embedding_features(self, x):
        out = self.extractor(x)
        out = out.view(out.size(0), -1)
        return out
model = MyNet().cuda()

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

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()

test_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
data = dset.ImageFolder(root="./data/val", transform=test_transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=200, shuffle=True,  num_workers=4)

acc = test(data_loader)
print("acc: %.4f" % (acc))
