import torch
from os.path import join
from tqdm import tqdm

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

