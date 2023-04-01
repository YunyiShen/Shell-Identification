import torch
from torch.utils.data import Dataset
import json
from PIL import Image

class TurtlePair(Dataset): 
    '''
    dataset class, take a json file, the list is a dictionary with img1, img2, label
        items are pairs of images and labels 
        for whether they are the same turtle or not
    '''
    def __init__(self, data_file, transform=None):
        with open(data_file, 'r') as jsonfile:
            self.data = json.load(jsonfile)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image1 = Image.open(self.data[idx][0]).convert('RGB')
        image2 = Image.open(self.data[idx][1]).convert('RGB')
        label = torch.tensor(self.data[idx][2])
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label

