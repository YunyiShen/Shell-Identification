import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from ../utils/utils import *


#train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

'''
File structure assumption:

dataset/
    train.txt
    val.txt
    train/
        green/
            green_001.jpg
            green_002.jpg
            ...
        hawksbill/
            hawksbill_001.jpg
            hawksbill_002.jpg
            ...
        loggerhead/
            loggerhead_001.jpg
            loggerhead_002.jpg
            ...
    val/
        green/
            green_101.jpg
            green_102.jpg
            ...
        hawksbill/
            hawksbill_101.jpg
            hawksbill_102.jpg
            ...
        loggerhead/
            loggerhead_101.jpg
            loggerhead_102.jpg
            ...
    test/
        green/
            green_201.jpg
            green_202.jpg
            ...
        hawksbill/
            hawksbill_201.jpg
            hawksbill_202.jpg
            ...
        loggerhead/
            loggerhead_201.jpg
            loggerhead_202.jpg
            ...

'''



'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
'''

transform_train = create_transform(384, is_training = True, auto_augment = "rand-m9-mstd0.5")


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
])


# Define the dataset class for loading turtle photos
class TurtleDataset(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'r') as f:
            self.image_files = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(image_file.split('/')[-2]) # see file structure
        return image, label


def train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs = 10):
    best_val_acc = 0
    model.to(device)
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            #print(images.shape)
            images, labels = images.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            total = 0
            correct = 0
            for images, labels in val_loader:
                images, labels = images.to(device), torch.tensor(labels).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")
        if accuracy > best_val_acc:
            torch.save(model.state_dict(), 'turtle_classifier.pth')
            best_val_acc = accuracy

    
def main(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    #model = timm.create_model('vit_base_patch16_224', num_classes = 10)
    model = timm.create_model('vit_base_patch16_384', num_classes = 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    

    # Set up datasets and dataloaders
    train_loader = DataLoader( TurtleDataset(data_file='./dataset/train.txt', transform=transform_train), batch_size = 5)
    val_loader = DataLoader( TurtleDataset(data_file='./dataset/val.txt', transform=transform_test), batch_size = 5)
    train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs)



