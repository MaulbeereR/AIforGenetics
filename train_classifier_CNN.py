import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import torch.optim as optim
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from PIL import Image

from common import *


def indices_to_one_hot(labels, num_classes):
    label_count = labels.shape[0]

    one_hot_labels = np.zeros((label_count, num_classes), dtype='int')
    for i in range(label_count):
        label = labels[i]
        one_hot_labels[i][label] = 1.0

    return one_hot_labels


class FACS_dataset(Dataset):
    def __init__(self, data, labels):
        x = np.expand_dims(data, 1)
        # x = np.expand_dims(x, 3)

        labels = labels.astype('int')

        y = indices_to_one_hot(labels, 2)

        print('x.shape: ', x.shape)
        print('y.shape: ', y.shape)

        self.data = x.astype('float')
        self.labels = y.astype('int')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = (self.data[idx])
        label = (self.labels[idx])

        return sample, label


class cnn_model_2D(nn.Module):
    def __init__(self):
        super(cnn_model_2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, [1, ch_count])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.pool1 = nn.AvgPool2d([1,2])
        self.conv2 = nn.Conv2d(12, 24, [1, 1])
        # self.pool2 = nn.AvgPool2d([1,2])
        # self.conv3 = nn.Conv2d(16, 32, [1,2])

        self.pool3 = nn.AvgPool2d([cell_count, 1])

        # self.fc1 = nn.Linear(17280, 600)
        self.fc1 = nn.Linear(24, 2)
        # self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        # y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu(y)
        # y = self.pool2(y)
        # y = self.conv3(y)
        y = self.pool3(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        # y = self.relu(y)
        # y = self.fc2(y)
        y = self.sigmoid(y)
        # y = self.relu5(y)
        return y


def process_raw_data(cell_count, input_path, metadata, sample_count):
    input_path += 'fcs_data_V2/'

    input_filelist = os.listdir(input_path)

    all_data_list = []
    all_labels_list = []

    for f_idx, fcs_filename in enumerate(input_filelist):

        if f_idx <= sample_count:

            label = 0
            mouse_sex = metadata.iloc[f_idx]['sex']
            if mouse_sex == 'F':
                label = 1

            print(fcs_filename)

            cell_data = read_FCS(input_path + fcs_filename)
            print(cell_data.shape)

            if cell_data.shape[0] > cell_count:
                trimmed_data = cell_data[0:cell_count, :]
                if trimmed_data.shape == (cell_count, 6):
                    all_data_list.append(trimmed_data)
                    all_labels_list.append(label)

    all_data = np.stack(all_data_list)
    all_labels = np.stack(all_labels_list)

    return all_data, all_labels


model_type = '2D'
# model_type = 'hist_mlp'

input_path = '../'
metadata = pd.read_csv(input_path + 'filtered_FCS_data_2024-03-07_V2.csv')

data_2D_filename = '../dataset/all_data_V4.npy'
label_filename = '../dataset/all_labels_V4.npy'

num_classes = 2
cell_count = 30000
ch_count = 6

do_process_data = False

if do_process_data:
    all_data, all_labels = process_raw_data(cell_count=cell_count, input_path=input_path, metadata=metadata,
                                            sample_count=10000)
    np.save(data_2D_filename, all_data)
    np.save(label_filename, all_labels)

raw_data = np.load('../dataset/all_data_V4.npy', allow_pickle=True)
labels_sex = np.load('../dataset/all_labels_V4.npy', allow_pickle=True)

print('raw_data.shape: ', raw_data.shape)
print('labels_sex.shape: ', labels_sex.shape)

raw_labels = labels_sex

female_indices = list(np.where(raw_labels == 1)[0])
print('female samples before balancing: ', len(female_indices))

male_indices = list(np.where(raw_labels == 0)[0])
print('male samples before balancing: ', len(male_indices))

female_indices_balanced_list = female_indices
male_indices_balanced_list = random.sample(male_indices, len(female_indices_balanced_list))
print('male samples after balancing: ', len(male_indices_balanced_list))

print(raw_data[female_indices_balanced_list].shape)
print(raw_data[male_indices_balanced_list].shape)

balanced_data = np.concatenate(
    [raw_data[male_indices_balanced_list].squeeze(), raw_data[female_indices_balanced_list].squeeze()], axis=0)
balanced_labels = np.concatenate(
    [raw_labels[male_indices_balanced_list].squeeze(), raw_labels[female_indices_balanced_list].squeeze()], axis=0)

print('balanced_data.shape: ', balanced_data.shape)
print('balanced_labels.shape: ', balanced_labels.shape)

if model_type == '2D':
    # ******************************
    # TASK: sex, MODEL_TYPE: 2D

    sample_count = balanced_data.shape[0]

    x_train, x_test, y_train, y_test = train_test_split(balanced_data, balanced_labels, test_size=0.1, shuffle=True)

    training_data = FACS_dataset(x_train, y_train)
    test_data = FACS_dataset(x_test, y_test)

    model = cnn_model_2D().cuda()
    max_epochs = 50
    LR = 0.01
    batch_size = 16

    summary(model, (batch_size, 1, cell_count, ch_count), device='cuda')

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)

print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=0.0001)

best_accuracy = 0

for epoch in range(max_epochs):

    model.train()
    batch_loss = 0

    print('========== EPOCH ', epoch, '===========')

    for x, y in train_dataloader:
        # print(x)
        # print(y)

        x = x.cuda()
        y = torch.tensor(y).cuda().float()

        optimizer.zero_grad()
        predict_y = model(x.float())

        # print(y)
        # print(predict_y, ':', y)

        # print(output)

        loss = criterion(predict_y, y)  # criterion(output, y)

        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

    print('train loss: ', batch_loss / len(train_dataloader))

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer

    total = 0
    correct = 0
    for x_t, y_t in test_dataloader:
        x_t = x_t.cuda()
        y_t = y_t.cuda().float()

        predict_y = model(x_t.float())

        loss = criterion(predict_y, y_t)
        valid_loss += loss.item()

        predict_y = torch.argmax(predict_y, dim=-1)
        correct += (predict_y == torch.argmax(y_t)).float().sum()

        total += y_t.size(0)

    print('validation loss: ', valid_loss / len(test_dataloader))
    accuracy = correct / total
    print("Accuracy = {}".format(accuracy))

    if accuracy > best_accuracy:
        torch.save(model, '../models/model_2D_V1.pkl')
        print('Accuracy increased - saving model')
        best_accuracy = accuracy

all_correct_num = 0
all_sample_num = 0
model.eval()

total = 0
correct = 0

best_model = torch.load('../models/model_2D_V1.pkl')

for i, (test_x, test_y) in enumerate(test_dataloader):
    test_x = test_x.cuda()
    test_y = test_y.cuda()

    predict_y = best_model(test_x.float())
    predict_y = torch.argmax(predict_y, dim=-1)

    # print(predict_y, ':', test_y)

    total += test_y.size(0)
    correct += (predict_y == torch.argmax(test_y)).float().sum()

accuracy = correct / total

print("Accuracy = {}".format(accuracy))

print("Model finished training")
