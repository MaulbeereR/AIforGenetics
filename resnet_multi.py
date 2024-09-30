import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


train_size = 0.8
val_size = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = np.load('../output/dataset/new_labels.npy', allow_pickle=True)
print('labels shape: ', labels.shape)


class NPYDatasetWithAuxiliary(Dataset):
    def __init__(self, directory, labels):
        self.directory = directory
        self.labels = labels
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        sex_label = int(self.filenames[idx].split('_')[1][0])
        return torch.tensor(data, dtype=torch.float32), label, sex_label

    def get_data_size(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        return data.shape


class MultiTaskResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc_primary = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 99)
        )

        self.fc_auxiliary = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet(x)
        primary_output = self.fc_primary(x)
        auxiliary_output = self.fc_auxiliary(x)
        return primary_output, auxiliary_output


def train_model_with_auxiliary_loss(model, criterion, criterion_sex, auxiliary_ratio, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    train_losses = []
    val_losses = []
    f1_scores = []
    auroc_scores = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        count = 0
        all_preds = []
        all_labels = []

        for inputs, labels, sex_labels in train_loader:

            inputs, labels, sex_labels = inputs.to(device), labels.to(device), sex_labels.to(device)
            optimizer.zero_grad()

            outputs, sex_outputs = model(inputs)

            loss = 0
            for i in range(99):
                alpha = 1 - np.mean(labels[:, i].cpu().numpy())
                loss += ops.sigmoid_focal_loss(outputs[:, i], labels[:, i], alpha=alpha, gamma=2.0, reduction='mean')
            loss /= 99

            sex_loss = criterion_sex(sex_outputs, sex_labels)

            loss = loss + auxiliary_ratio * sex_loss

            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            running_loss += loss.item()
            count += 1

        epoch_loss = running_loss / count
        train_losses.append(epoch_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_auroc = roc_auc_score(all_labels, all_preds, average='weighted')

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1 Score: {train_f1:.4f}, Train AUROC: {train_auroc:.4f}')

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        count = 0
        all_preds = []
        all_labels = []

        for inputs, labels, sex_labels in val_loader:
            inputs, labels, sex_labels = inputs.to(device), labels.to(device), sex_labels.to(device)
            with torch.no_grad():
                outputs, sex_outputs = model(inputs)
                loss = 0
                for i in range(99):
                    alpha = 1 - np.mean(labels[:, i].cpu().numpy())
                    loss += ops.sigmoid_focal_loss(outputs[:, i], labels[:, i], alpha=alpha, gamma=2.0, reduction='mean')
                loss /= 99
                sex_loss = criterion_sex(sex_outputs, sex_labels)
                loss = loss + auxiliary_ratio * sex_loss

                preds = (torch.sigmoid(outputs) > 0.5).float()

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
        val_auroc = roc_auc_score(all_labels, all_preds, average='weighted')

        f1_scores.append(val_f1)
        auroc_scores.append(val_auroc)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f'Best Val F1: {best_f1:.4f}')
    print(f'Best Val AUROC: {max(auroc_scores):.4f}')
    return model, train_losses, val_losses, f1_scores, auroc_scores


def evaluate_on_validation(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels, sex_labels in val_loader:
        inputs, labels, sex_labels = inputs.to(device), labels.to(device), sex_labels.to(device)
        with torch.no_grad():
            outputs, sex_outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    precision_per_label = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_label = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_label = f1_score(all_labels, all_preds, average=None, zero_division=0)
    accuracy_per_label = np.mean(all_preds == all_labels, axis=0)
    auroc_per_label = roc_auc_score(all_labels, all_preds, average=None)

    metrics = {
        'Label': list(range(1, 100)),
        'Precision': precision_per_label,
        'Recall': recall_per_label,
        'F1 Score': f1_per_label,
        'Accuracy': accuracy_per_label,
        'AUROC': auroc_per_label
    }

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('../output/meeting_result/label_metrics_multi.csv', index=False)


dataset = NPYDatasetWithAuxiliary(f'../output/tensor_data/', labels)
print('dataset length: ', len(dataset))
print('sample size: ', dataset.get_data_size(0))

train_dataset, val_dataset = train_test_split(dataset, test_size=val_size, random_state=24)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MultiTaskResNet(pretrained=True).to(device)
criterion = nn.BCEWithLogitsLoss()
criterion_sex = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = model.to(device)
trained_model, train_losses, val_losses, f1_scores, auroc_scores = train_model_with_auxiliary_loss(
    model, criterion, criterion_sex, 0.1, optimizer, exp_lr_scheduler, num_epochs=25)
evaluate_on_validation(trained_model, val_loader)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Per Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(auroc_scores, label='Validation AUROC')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Validation AUROC Per Epoch')
plt.legend()
plt.show()
