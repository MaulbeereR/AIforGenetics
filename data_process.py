import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import flowkit as fk
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

labels_sex = np.load('../output/dataset/all_labels_V4.npy', allow_pickle=True)
print('labels_sex.shape: ', labels_sex.shape)

female_indices = list(np.where(labels_sex == 1)[0])
print('female samples: ', len(female_indices))

male_indices = list(np.where(labels_sex == 0)[0])
print('male samples: ', len(male_indices))


def combined_images():
    image_size = (80, 80)
    grid_size = (3, 5)
    output_image_size = (image_size[0] * grid_size[1], image_size[1] * grid_size[0])
    pairs = list(combinations(range(6), 2))
    num_datasets = 2525

    for idx in range(num_datasets):
        print(idx)
        output_image = Image.new('L', output_image_size, 255)

        for position, (i, j) in enumerate(pairs):
            filename = f'../output/80_80_output/data{idx}_{i}_{j}.png'
            img = Image.open(filename)

            if img.mode != 'L':
                img = img.convert('L')

            x = (position % grid_size[1]) * image_size[0]
            y = (position // grid_size[1]) * image_size[1]

            output_image.paste(img, (x, y))

        output_image.save(f'../output/conbined_image/data{idx}_{labels_sex[idx]}.png')


def combined_list():
    pairs = list(combinations(range(6), 2))
    num_datasets = 2525

    for idx in range(num_datasets):
        img_list = []
        print(idx)
        for (i, j) in pairs:
            filename = f'../output/80_80_output/data{idx}_{i}_{j}.png'
            img = Image.open(filename)
            img_tensor = ToTensor()(transforms.Grayscale()(img))

            img_list.append(img_tensor)

        print(len(img_list))
        print(img_list[1].size)

        data = torch.cat(img_list, dim=0)
        data_np = data.cpu().numpy()
        print(data_np.shape)

        np.save(f'../output/tensor_data/data{idx}_{labels_sex[idx]}.npy', data_np)


# combined_images()
# combined_list()
