import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.io as visionio
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from os.path import join
from PIL import Image
from tqdm import tqdm


# class for datasets
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

class TurtleDiff(nn.Module):
    def __init__(self, backbone, hidden = 100,pretrained = True):
        super(TurtleDiff, self).__init__()
        self.backbone = timm.create_model(backbone, 
                                          pretrained=pretrained,
                                          num_classes=0)
        self.backbone_name = backbone
        self.fc = nn.Linear(self.backbone.num_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)

        # Freeze the parameters of the backbone if use pretrained
        if(pretrained):
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x1, x2):
        x1 = self.backbone(x1) 
        x2 = self.backbone(x2)
        # difference between the two image embeddings, one way to make sure symmetry
        x = torch.abs(x1 - x2) 
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

class TurtleDiffPool(nn.Module):
    def __init__(self, backbone, hidden = 100,pretrained = True):
        super(TurtleDiffPool, self).__init__()
        self.backbone = timm.create_model(backbone, 
                                          pretrained=pretrained,
                                          num_classes=0)
        self.backbone_name = backbone + "-pool"
        self.fc = nn.Linear(self.backbone.num_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)

        # Freeze the parameters of the backbone if use pretrained
        if(pretrained):
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x1, x2):
        x1 = self.backbone(x1) 
        x2 = self.backbone(x2)
        # get two way difference, then pool them after nonlinearity
        x1p = x1 - x2
        x2p = x2 - x1
        x1p = self.fc(x1p)
        x1p = nn.ReLU()(x1p)
        x2p = self.fc(x2p)
        x2p = nn.ReLU()(x2p)
        x = (x1p+x2p)/2
        x = self.fc2(x)
        return x

class TurtleDiffConv(nn.Module):
    def __init__(self, hidden = 100):
        super(TurtleDiffConv, self).__init__()
        self.backbone = timm.create_model('resnet50', 
                                          pretrained=True,
                                          num_classes=0)
        self.backbone_name = 'resnet50-conv'

        self.fc = nn.Linear(self.backbone.num_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)

        # Freeze the parameters of the backbone if use pretrained
        for name, param in self.backbone.named_parameters():
            if name.startswith("layer4.2"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x1, x2):
        x1 = self.backbone(x1) 
        x2 = self.backbone(x2)
        # difference between the two image embeddings, one way to make sure symmetry
        x = torch.abs(x1 - x2) 
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

def train(model, optimizer, criterion, 
          train_loader, 
          val_loader, device, 
          num_epochs = 10, 
          save_path = "./"):
    best_val_acc = 0
    model.to(device)
    loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for images1, images2, labels in tqdm(train_loader):
            #print(images.shape)
            images1, images2, labels = images1.to(device), \
                                       images2.to(device), \
                                       labels.to(device)
            optimizer.zero_grad()
            outputs = model(images1, images2)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            total = 0
            correct = 0
            for images1, images2, labels in tqdm(val_loader):
                images1, images2, labels = images1.to(device), \
                                           images2.to(device), \
                                           labels.to(device)
                outputs = model(images1, images2)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")
        if accuracy > best_val_acc:
            torch.save(model.state_dict(), 
                       join(save_path, model.backbone_name + 
                            '_turtle_identifier.pth')
                       )
            best_val_acc = accuracy
    return loss_list

