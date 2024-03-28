import numpy as np
import os

from common import *

input_path = '../fcs_data_V2/'

input_filelist = os.listdir(input_path)
output_path = 'output/'

all_data_list = []

cell_count = 30000

sample_count = 3

for f_idx, fcs_filename in enumerate(input_filelist):

    if f_idx <= sample_count:
        print(fcs_filename)

        cell_data = read_FCS(input_path + fcs_filename)

        ch_1 = 'B220'
        ch_2 = 'CD3'
        fig = plot_scatter_logicle(cell_data, ch_1, ch_2)
        output_filename = str(f_idx).zfill(5) + '_' + ch_1 + '_' + ch_2 + '.png'
        fig.savefig(output_path + output_filename)

        ch_1 = 'IgM'
        ch_2 = 'IgD'
        fig = plot_scatter_logicle(cell_data, ch_1, ch_2)
        output_filename = str(f_idx).zfill(5) + '_' + ch_1 + '_' + ch_2 + '.png'
        fig.savefig(output_path + output_filename)

        all_data_list.append(cell_data[0:cell_count, :])

all_data = np.vstack(all_data_list)

print('all_data.shape: ', all_data.shape)
