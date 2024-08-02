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


scattering_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
antibody_channels = ['IgM', 'IgD', 'B220', 'CD44', 'CD3', 'CD4']

raw_data = np.load('../output/new/dataset/new_all_data.npy', allow_pickle=True)
labels = np.load('../output/new/dataset/new_all_labels.npy', allow_pickle=True)
output_path = '../output/new/tensor_data/'

print('raw_data.shape: ', raw_data.shape)  # shape: (2491, 30000, 6)
print('labels.shape: ', labels.shape)      # shape: (2491, 17)


def generate_tensored_data(data, n, resolution=80):
    pairs = list(combinations(range(6), 2))
    img_list = []

    for antibody_x, antibody_y in pairs:
        x = data[:, antibody_x]
        y = data[:, antibody_y]

        bins = (resolution, resolution)

        hist_data, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)

        z = interpn((0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])), hist_data,
                    np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

        z[np.isnan(z)] = 0.0

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots(figsize=(1, 1), dpi=resolution)
        ax.scatter(x, y, c=z, s=0.2, cmap='gray')
        ax.set_axis_off()

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(img_array).convert('L')
        img_tensor = ToTensor()(img)

        img_list.append(img_tensor)
        plt.close(fig)

    data_tensor = torch.cat(img_list, dim=0)
    return data_tensor


for num in range(len(raw_data)):
    tensored_data = generate_tensored_data(raw_data[num], num)
    np.save(f'{output_path}/data{num}.npy', tensored_data.numpy())
    print(f'Dataset {num} processed with shape {tensored_data.size()}')
