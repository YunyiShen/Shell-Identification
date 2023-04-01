import torch
import timm
import torch.nn as nn
from tqdm import tqdm


# class for datasets
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
        '''
        x = x1 - x2
        x = self.fc(x)
        x = torch.abs(x) # equivalent to the above
        '''
        x = self.fc2(x)
        
        return x
    

class TurtleDiffConvPool(nn.Module):
    def __init__(self, hidden = 100):
        super(TurtleDiffConvPool, self).__init__()
        
        self.backbone = timm.create_model('resnet50', 
                                          pretrained=True,
                                          num_classes=0)
        
        self.backbone_name = 'resnet50-conv-pool'
        
        self.fc = nn.Linear(self.backbone.num_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)

        for name, param in self.backbone.named_parameters():
            if name.startswith("layer4.2.conv3"):
                param.requires_grad = True
            else:
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
        '''
        x = x1 - x2
        x = self.fc(x)
        x = torch.abs(x) # equivalent to the above
        '''
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
