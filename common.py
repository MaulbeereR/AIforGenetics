import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import flowkit as fk
from matplotlib.patches import Polygon
# from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image

from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib.colors import LogNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import torch.optim as optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import random
import json
import os
import math

scattering_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
antibody_channels = ['IgM', 'IgD', 'B220', 'CD44', 'CD3', 'CD4']

polydata_cells = [
    (12000, 6000),
    (120000, 8000),
    (200000, 100000),
    (240000, 160000),
    (230000, 230000),
    (150000, 250000),
    (90000, 250000),
    (70000, 230000),
    (10000, 25000),
    (12000, 8000)
]


# cleaned up
def read_FCS(fcs_filename):
    sample = fk.Sample(fcs_filename)

    compensation_matrix = sample.metadata['spill']
    sample.apply_compensation(compensation_matrix)

    g_strat = fk.GatingStrategy()

    # define the dimensions (parameters) for gating
    dim_fsc_a = fk.Dimension('FSC-A')
    dim_ssc_a = fk.Dimension('SSC-A')
    dim_fsc_h = fk.Dimension('FSC-H')

    poly_gate_cells = fk.gates.PolygonGate('poly_cells', dimensions=[dim_fsc_a, dim_ssc_a], vertices=polydata_cells)

    g_strat.add_gate(poly_gate_cells, gate_path=('root',))

    gs_result = g_strat.gate_sample(sample)

    print(gs_result.report)

    df_all = sample.as_dataframe(source='raw', subsample=False)

    bool_singlecells = gs_result.get_gate_membership('poly_cells')

    logicle_xform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)
    sample.apply_transform(logicle_xform)

    fluoro_channels = []

    print(sample.pnn_labels)
    print(sample.pns_labels)

    for antibody_name in antibody_channels:
        try:
            idx = sample.pns_labels.index(antibody_name)
            fluoro_channels.append(sample.pnn_labels[idx])
        except ValueError:
            continue

    # ?
    df_fluoro = sample.as_dataframe(source='xform', subsample=False)[fluoro_channels]

    data_cells = np.array(df_fluoro[bool_singlecells])

    if data_cells.size > 0:
        print('min: ', data_cells.flatten().min())
        print('max: ', data_cells.flatten().max())

    print('data_cells.shape: ', data_cells.shape)

    return(data_cells)


def plot_scatter_logicle(data, antibody_x, antibody_y):
    ch_x_idx = antibody_channels.index(antibody_x)
    ch_y_idx = antibody_channels.index(antibody_y)

    print(ch_x_idx)

    x = data[:, ch_x_idx]
    y = data[:, ch_y_idx]

    print('x.shape: ', x.shape)
    print('y.shape: ', y.shape)

    bins = [512, 512]

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    norm = Normalize(vmin=np.min(z), vmax=np.max(z) * 0.25)

    ax.scatter(x, y, c=z, s=0.2, norm=norm)
    # ax.grid()

    ax.set_xlim(xmin=-0.5, xmax=1)
    ax.set_ylim(ymin=-0.5, ymax=1)

    # axis_ticks = intervals

    # ax.set_xticks(axis_ticks)
    # ax.set_yticks(axis_ticks)

    ax.set_xlabel(antibody_x, fontsize=20)
    ax.set_ylabel(antibody_y, fontsize=20)

    return fig
