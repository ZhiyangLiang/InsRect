import torch
import numpy as np
from bisect import bisect_left
import pdb
from PIL import Image

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None):

        self.data = np.load('../data/tiny_images/300K_random_images.npy')
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index, :, :, :]
        img =Image.fromarray(np.uint8(img))
        img = self.transform(img)
        return img, -1
        
    def __len__(self):
        return self.data.shape[0]

