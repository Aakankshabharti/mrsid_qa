import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import re

class ImageMOSDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # read the CSV file
        self.data = pd.read_csv(csv_path)
        self.transforms = transform

        # flip labels here (0 -> 1, 1 -> 0)
        self.data['labels_7'] = self.data['labels_7'].apply(lambda x: 1 if x == 0 else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # image paths
        img_path1 = self.data.iloc[idx]['image1']
        if '_scene1' in img_path1:
             img_path1 = re.sub(r'_scene1\.bmp$', '.bmp', img_path1) 
        if '_scene4' in img_path1:
             img_path1 = re.sub(r'_scene4\.bmp$', '.bmp', img_path1) 
        img_path2 = self.data.iloc[idx]['image2']
        if '_scene1' in img_path2:
             img_path2 = re.sub(r'_scene1\.bmp$', '.bmp', img_path2)
        if '_scene4' in img_path2:
             img_path2 = re.sub(r'_scene4\.bmp$', '.bmp', img_path2) 

        # load images
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')

        # apply transforms
        image1 = self.transforms(image1)
        image2 = self.transforms(image2)

        # label after flipping
        label = int(self.data.iloc[idx]['labels_7'])

        return image1, image2, label, img_path1, img_path2
