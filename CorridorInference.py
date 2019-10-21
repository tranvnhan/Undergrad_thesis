import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pdb
from onehot import onehot
import torch
import numpy as np
import PIL

data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# dataset_dir = 'Datasets/2-UMichigan-corridor/'
# root_dir = 'Datasets/3-TQB-library-corridor/'
# root_dir = 'Datasets/Train-Validation-Test/'
# root_dir = 'Datasets/CMU-corridor-train-validation/'
# root_dir = 'Datasets/6-only-one-corridor/'
root_dir = 'Datasets/xyz/'


class CorridorDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset_dir = root_dir + dataset
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.dataset_dir + '/raw-image/'))

    def __getitem__(self, idx):
        img_name = os.listdir(self.dataset_dir + '/raw-image/')[idx]
        imgA = cv2.imread(self.dataset_dir + '/raw-image/' + img_name)  # input image
        imgA = cv2.resize(imgA, (96, 96))

        if self.transform:
            imgA = self.transform(imgA)

        item = {'A': imgA, 'N': img_name}  # dictionary in Python
        return item


corridor = {x: CorridorDataset(x, transform=data_transforms[x])
            for x in ['test']}
dataloader = {x: DataLoader(corridor[x], batch_size=1, shuffle=False, num_workers=1)
              for x in ['test']}

# if __name__ == '__main__':
#     for batch in trainloader:
#         break
