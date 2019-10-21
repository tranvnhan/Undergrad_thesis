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
    'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
}

# dataset_dir = 'Datasets/2-UMichigan-corridor/'
# dataset_dir = 'Datasets/3-TQB-library-corridor/'
# root_dir = 'Datasets/Train-Validation-Test/'
root_dir = 'Datasets/CMU-augment-corridor-train-validation/'
# root_dir = 'Datasets/CMU-corridor-train-validation/'


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

        imgB = cv2.imread(self.dataset_dir + '/ground-truth/' + img_name, 0)  # label image
        imgB = cv2.resize(imgB, (96, 96))            
        imgB = imgB / 255
        imgB = imgB.astype('uint8')  # convert to binary image
        imgB = onehot(imgB, 2)
        imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)

        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)   # (2, 160, 160)

        item = {'A': imgA, 'B': imgB}  # dictionary in Python
        return item

corridor = {x: CorridorDataset(x, transform=data_transforms[x])
            for x in ['train', 'val']}
dataloader = {x: DataLoader(corridor[x], batch_size=4, shuffle=True, num_workers=4)  # original batch size = 4
              for x in ['train', 'val']}

# if __name__ == '__main__':
#     for batch in trainloader:
#         break
