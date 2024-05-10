import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt


train_size = 0.7
val_size = 0.15
test_size = 0.15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NPYDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        label = int(self.filenames[idx].split('_')[1][0])
        return torch.tensor(data, dtype=torch.float32), label

    def get_data_size(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        return data.shape


dataset = NPYDataset(f'../output/tensor_data/')
print('dataset length: ', len(dataset))
print('sample size: ', dataset.get_data_size(0))

train_val, test_dataset = train_test_split(dataset, test_size=test_size, random_state=24)
train_dataset, val_dataset = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=24)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = model.to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()

        # Training phase
        running_loss = 0.0
        count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1

            running_loss += loss.item()

        epoch_loss = running_loss / count
        train_losses.append(epoch_loss)

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        count = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_loss += loss.item()
                count += 1

        epoch_val_loss = val_loss / count
        val_losses.append(epoch_val_loss)

        val_acc = val_corrects.double() / len(val_dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f'Best Val Acc: {best_acc:.4f}')
    return model, train_losses, val_losses


trained_model, train_losses, val_losses = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)


def evaluate_model(model, test_loader):
    model.eval()
    test_corrects = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    test_acc = test_corrects.double() / len(test_dataset)
    print(f'Test Accuracy: {test_acc:.4f}')


evaluate_model(trained_model, test_loader)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Per Epoch')
plt.legend()
plt.show()
