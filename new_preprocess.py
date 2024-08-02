import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib.colors import LogNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision import transforms
import random
import json
import os
import math

from itertools import combinations
import common

unique_pathways = ['R-HSA-141430', 'R-HSA-3270619', 'R-HSA-912526', 'R-HSA-9617324', 'R-HSA-174048', 'R-HSA-936837', 'R-HSA-9013149', 'R-HSA-179409', 'R-HSA-9687136', 'R-HSA-9679191', 'R-HSA-168638', 'R-HSA-8853884', 'R-HSA-176412', 'R-HSA-176407', 'R-HSA-1266695', 'R-HSA-71384', 'R-HSA-9020558']
input_path = '../fcs_data_V2/'
metadata = pd.read_csv('../filtered_FCS_data_2024-03-07_V4_updated.csv')


def process_raw_data(cell_count, input_path, metadata):
    all_data_list = []
    all_labels_list = []

    for index, row in metadata.iterrows():
        file_id = row['file_id']
        pathways = row['pathways']

        print(index, file_id, pathways)

        label = [0] * 17
        pathways_list = pathways.split('|')
        for pw in pathways_list:
            if pw in unique_pathways:
                label[unique_pathways.index(pw)] = 1
        print(label)

        cell_data = common.read_FCS(input_path + file_id)
        if cell_data.shape[0] > cell_count:
            trimmed_data = cell_data[0:cell_count, :]
            if trimmed_data.shape == (cell_count, 6):
                all_data_list.append(trimmed_data)
                all_labels_list.append(label)

    all_data = np.stack(all_data_list)
    all_labels = np.stack(all_labels_list)
    return all_data, all_labels


def gen_data():
    data_filename = '../output/new/dataset/new_all_data.npy'
    label_filename = '../output/new/dataset/new_all_labels.npy'

    cell_count = 30000

    data, labels = process_raw_data(cell_count, input_path, metadata)
    np.save(data_filename, data)
    np.save(label_filename, labels)


data = np.load('../output/new/dataset/new_all_data.npy', allow_pickle=True)
labels = np.load('../output/new/dataset/new_all_labels.npy', allow_pickle=True)

print('raw_data.shape: ', data.shape)  # (2491, 30000, 6)
print('labels_sex.shape: ', labels.shape)  # (2491, 17)


