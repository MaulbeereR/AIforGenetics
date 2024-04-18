import numpy as np
import matplotlib.pyplot as plt
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
output_path = '../output/'

print('raw_data.shape: ', raw_data.shape)
# shape: (2525, 30000, 6)


def plot_heatmap(data, num, antibody_x, antibody_y, resolution=(80, 80)):
    x = data[:, antibody_x]
    y = data[:, antibody_y]

    bins = resolution

    inches_per_bin = 10
    fig, ax = plt.subplots()
    fig.set_size_inches(resolution[0] / inches_per_bin, resolution[1] / inches_per_bin)

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    z[np.where(np.isnan(z))] = 0.0

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    norm = Normalize(vmin=np.min(z), vmax=np.max(z) * 0.25)

    ax.scatter(x, y, c=z, s=0.2, norm=norm)

    ax.set_axis_off()

    output_filename = 'data' + str(num) + '_' + str(antibody_x) + '_' + str(antibody_y) + '.png'
    dpi = inches_per_bin * resolution[0] / 80
    fig.savefig(output_path + output_filename, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)


pairs = list(combinations(range(6), 2))
for num in range(2):
    for i, j in pairs:
        plot_heatmap(raw_data[num], num, i, j)

