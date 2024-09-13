import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


train_size = 0.7
val_size = 0.15
test_size = 0.15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = np.load('../output/new/dataset/new_all_labels.npy', allow_pickle=True)
print('labels shape: ', labels.shape)


class NPYDataset(Dataset):
    def __init__(self, directory, labels):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.npy')]
        self.labels = labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = label[0:3]
        return torch.tensor(data, dtype=torch.float32), label

    def get_data_size(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        return data.shape


dataset = NPYDataset(f'../output/new/tensor_data/', labels)
print('dataset length: ', len(dataset))
print('sample size: ', dataset.get_data_size(0))

train_val, test_dataset = train_test_split(dataset, test_size=test_size, random_state=24)
train_dataset, val_dataset = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=24)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = model.to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()

        # Training phase
        running_loss = 0.0
        count = 0
        train_corrects = np.zeros(3)
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss = ops.sigmoid_focal_loss(outputs, labels, alpha=0.8, gamma=2.0, reduction='mean')

            loss = 0
            for i in range(3):
                alpha = 1 - np.mean(labels[:, i].cpu().numpy())
                loss += ops.sigmoid_focal_loss(outputs[:, i], labels[:, i], alpha=alpha, gamma=2.0, reduction='mean')
            loss /= 3

            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            train_corrects += (preds == labels).sum(dim=0).cpu().numpy()
            count += 1
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            running_loss += loss.item()

        epoch_loss = running_loss / count
        train_losses.append(epoch_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, train Precision: {train_precision:.4f}, train Recall: {train_recall:.4f}, train F1 Score: {train_f1:.4f}')

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = np.zeros(3)
        count = 0
        all_preds = []
        all_labels = []

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                # loss = criterion(outputs, labels)
                loss = ops.sigmoid_focal_loss(outputs, labels, alpha=0.8, gamma=2.0, reduction='mean')

                preds = (torch.sigmoid(outputs) > 0.5).float()

                val_corrects += (preds == labels).sum(dim=0).cpu().numpy()
                val_loss += loss.item()
                count += 1

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_val_loss = val_loss / count
        val_losses.append(epoch_val_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}')

        # Update best model weights
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f'Best Val F1 Score: {best_f1:.4f}')
    return model, train_losses, val_losses


trained_model, train_losses, val_losses = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)


def evaluate_model(model, test_loader):
    model.eval()
    test_corrects = np.zeros(3)
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_corrects += (preds == labels).sum(dim=0).cpu().numpy()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(np.sum(all_preds, axis=0))
    print(np.sum(all_labels, axis=0))

    print(all_preds.shape)
    print(all_labels.shape)

    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

    precision_per_label = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_label = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_label = f1_score(all_labels, all_preds, average=None, zero_division=0)
    accuracy_per_label = np.mean(all_preds == all_labels, axis=0)

    metrics = {
        'Label': list(range(1, 4)),
        'Precision': precision_per_label,
        'Recall': recall_per_label,
        'F1 Score': f1_per_label,
        'Accuracy': accuracy_per_label
    }

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('../output/new/label_metrics.csv', index=False)


evaluate_model(trained_model, test_loader)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Per Epoch')
plt.legend()
plt.show()
