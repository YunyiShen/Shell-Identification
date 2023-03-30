import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv
from os.path import join

from ShellRec.training.shell_rec import TurtleDiff, TurtlePair, train


# Path: scripts/train_shell_rec.py
def main():
    transform_train = transforms.create_transform(384, is_training = True, 
                                   auto_augment = "rand-m9-mstd0.5")


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], 
                             std=[0.2290, 0.2240, 0.2250])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    #model = timm.create_model('vit_base_patch16_224', num_classes = 10)
    model = TurtleDiff('vit_base_patch16_384') # use a vit backbone
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    

    # Set up datasets and dataloaders
    train_loader = DataLoader( TurtlePair(data_file='./dataset/train.json', 
                                             transform=transform_train), 
                                             batch_size = 20)
    val_loader = DataLoader( TurtlePair(data_file='./dataset/val.json', 
                                           transform=transform_test), 
                                           batch_size = 20)
    train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs)

if __name__ == "__main__":
    main()
