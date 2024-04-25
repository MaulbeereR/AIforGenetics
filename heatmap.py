import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
import flowkit as fk
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

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
import torch.nn.functional as F
import random
import json
import os
import math
from itertools import combinations

scattering_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
antibody_channels = ['IgM', 'IgD', 'B220', 'CD44', 'CD3', 'CD4']

raw_data = np.load('../dataset/all_data_V4.npy', allow_pickle=True)
output_path = '../output/80_80_output/'

print('raw_data.shape: ', raw_data.shape)
# shape: (2525, 30000, 6)


def plot_heatmap(data, n, antibody_x, antibody_y, res):
    x = data[:, antibody_x]
    y = data[:, antibody_y]

    bins = (res, res)

    inches_per_bin = 10
    fig, ax = plt.subplots()
    fig.set_size_inches(res / inches_per_bin, res / inches_per_bin)

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)

    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    z[np.where(np.isnan(z))] = 0.0

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # norm = Normalize(vmin=np.min(z), vmax=np.max(z) * 0.25)
    # ax.scatter(x, y, c=z, s=0.2, norm=norm)

    ax.scatter(x, y, c=z, s=0.2, cmap='gray')

    ax.set_axis_off()

    output_filename = 'data' + str(n) + '_' + str(antibody_x) + '_' + str(antibody_y) + '.png'
    dpi = inches_per_bin
    fig.savefig(output_path + output_filename, pad_inches=0, dpi=dpi)
    plt.close(fig)


pairs = list(combinations(range(6), 2))
for num in range(len(raw_data)):
    for i, j in pairs:
        plot_heatmap(raw_data[num], num, i, j, 80)
    print(num)

