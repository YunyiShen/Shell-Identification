import torch
from os.path import join
from tqdm import tqdm
import torch.nn as nn


def inference(model, data_loader, device):
    results_list = []
    model.to(device)
    m = nn.Softmax(dim=1)
    with torch.no_grad():
            for images1, images2 in tqdm(data_loader):
                images1, images2 = images1.to(device), \
                                   images2.to(device)
                outputs = model(images1, images2)
                predicted = m(outputs.data)[:,1] # probability of being a match
                results_list.append(predicted.cpu().numpy().tolist())
    return results_list

def test_model(model, data_loader, device):
    results_list = []
    label_list = []
    model.to(device)
    m = nn.Softmax(dim=1)
    with torch.no_grad():
            for images1, images2, labels in tqdm(data_loader):
                images1, images2 = images1.to(device), \
                                   images2.to(device)
                outputs = model(images1, images2)
                predicted = m(outputs.data)[:,1] # probability of being a match
                label_list.append(labels.cpu().numpy().tolist())
                results_list.append(predicted.cpu().numpy().tolist())
    return {'predicted': results_list, 'labels': label_list}
