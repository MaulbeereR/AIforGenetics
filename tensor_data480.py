import numpy as np
from PIL import Image
import matplotlib
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
import gc

from itertools import combinations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

matplotlib.use('Agg')

scattering_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
antibody_channels = ['IgM', 'IgD', 'B220', 'CD44', 'CD3', 'CD4']

input_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'AIforGenetics')
output_folder = os.path.join(os.path.expanduser('~'), 'bigdisk', 'home','rongyu')

data_filename = os.path.join(os.path.expanduser(input_path), 'output/data.npy')
label_filename = os.path.join(os.path.expanduser(input_path), 'output/labels.npy')
idx_filename = os.path.join(os.path.expanduser(input_path), 'output/idx.npy')
raw_data = np.load(data_filename, allow_pickle=True)
labels = np.load(label_filename, allow_pickle=True)
idx = np.load(idx_filename, allow_pickle=True)
output_path = os.path.join(os.path.expanduser(output_folder), 'output/RGB_tensor_480')

print('raw_data.shape: ', raw_data.shape)  # shape: (2525, 30000, 6)
print('labels.shape: ', labels.shape)      # shape: (2525, 99)
print('idx.shape: ', idx.shape)            # shape: (2525, )


def generate_tensored_data(data, n, resolution=480):
    pairs = list(combinations(range(6), 2))
    img_list = []
    bins = (resolution, resolution)

    for antibody_x, antibody_y in pairs:
        x = data[:, antibody_x]
        y = data[:, antibody_y]        

        hist_data, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)

        z = interpn((0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])), hist_data,
                    np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

        z[np.isnan(z)] = 0.0

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots(figsize=(1, 1), dpi=resolution)
        ax.scatter(x, y, c=z, s=0.075, edgecolors='none')
        ax.set_axis_off()

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img = Image.fromarray(img_array).convert('L')
        img = Image.fromarray(img_array)
        img_tensor = ToTensor()(img).to(device)

        img_list.append(img_tensor)
        # fig.savefig(f'{output_path}/data{n}_{antibody_x}_{antibody_y}.png', pad_inches=0, dpi=resolution)
        plt.close('all')
        del fig, ax, img_array, img, img_tensor, hist_data, x_edges, y_edges, z, idx, x, y

    # data_tensor = torch.cat(img_list, dim=0).to(device)
    data_tensor = torch.stack(img_list).to(device)
    del img_list
    gc.collect()
    return data_tensor


for num in range(2300, 2525):
    tensored_data = generate_tensored_data(raw_data[num], num)
    np.save(f'{output_path}/data{idx[num]}.npy', tensored_data.cpu().numpy())
    torch.cuda.empty_cache()
    print(f'Dataset {num} processed with shape {tensored_data.size()} with index {idx[num]}')
    del tensored_data
    
