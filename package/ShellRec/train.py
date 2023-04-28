import torch
from os.path import join
from tqdm import tqdm
import torch.nn as nn

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

def train_embedding(model, optimizer, 
                    criterion, 
                    train_loader, 
                    val_loader, device, 
                    num_epochs = 10, 
                    save_path = "./"):
    best_val_loss = 10000
    model.to(device)
    loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for images1, images2, labels in tqdm(train_loader):
            #print(images.shape)
            labels = 2 * labels - 1 # Convert to -1, 1
            images1, images2, labels = images1.to(device), \
                                       images2.to(device), \
                                       labels.to(device)
            optimizer.zero_grad()
            embd1 = model(images1)
            embd2 = model(images2)
            loss = criterion(embd1, embd2, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            val_loss = 0
            for images1, images2, labels in tqdm(val_loader):
                images1, images2, labels = images1.to(device), \
                                           images2.to(device), \
                                           labels.to(device)
                labels = 2 * labels - 1 # Convert to -1, 1
                embd1 = model(images1)
                embd2 = model(images2)
                val_loss += criterion(embd1, embd2, labels)

            
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation loss: {val_loss:.2f}")
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 
                       join(save_path, model.backbone_name + 
                            '_turtle_embedding.pth')
                       )
            best_val_loss = val_loss
    return loss_list