import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from os.path import join

from ShellRec.training.shell_rec import TurtleDiff, TurtlePair, train
from ShellRec.data_utils.prepare_photos import get_img_graph, split_graph
import random


# Path: scripts/train_shell_rec.py
def main():
    random.seed(42)
    all_train = get_img_graph(path = "./dataset/BoxTurtle", drop_p=[0.99,0])
    split_graph(all_train, save_path = "./dataset")
    _ = get_img_graph(path = "./dataset/BoxTurtle_holdout", 
                  file_to_save = "./dataset/BoxTurtle_holdout.json")

    torch.hub.set_dir('./pretrained/') # place to save pretrained models
    transform_train = transforms.create_transform(384, is_training = True, 
                                   auto_augment = "rand-m9-mstd0.5")


    transform_test = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5], 
                             std=[0.5, 0.5, 0.5])
])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    #model = timm.create_model('vit_base_patch16_224', num_classes = 10)
    model = TurtleDiff('vit_base_patch16_384') # use a vit backbone
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    

    # Set up datasets and dataloaders
    train_loader = DataLoader( TurtlePair(data_file='./dataset/train.json', 
                                             transform=transform_train), 
                                             batch_size = 32)
    val_loader = DataLoader( TurtlePair(data_file='./dataset/val.json', 
                                           transform=transform_test), 
                                           batch_size = 32)
    loss_ = train(model, optimizer, criterion, 
                  train_loader, 
                  val_loader, device, 
                  num_epochs)

if __name__ == "__main__":
    main()
