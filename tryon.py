from common import *
import numpy as np
import pandas as pd

input_path = '../fcs_data_V2/'
input_filelist = os.listdir(input_path)
print(np.array(input_filelist).shape)

unique_pnn_label_sets = set()

for f_idx, fcs_filename in enumerate(input_filelist):
    sample = fk.Sample(input_path + fcs_filename)
    unique_pnn_label_sets.add(frozenset(sample.pnn_labels))

unique_pnn_label_sets = [set(labels) for labels in unique_pnn_label_sets]
print("Unique sets of PnN labels:", unique_pnn_label_sets)


metadata = pd.read_csv('../filtered_FCS_data_2024-03-07_V2.csv')
tot_pathway = []

for idx in range(2822):
    pathways = metadata.iloc[idx]['pathways']
    sep_pathway = pathways.split('|')
    for pathway in sep_pathway:
        tot_pathway.append(pathway)

tot_pathway = np.unique(np.array(tot_pathway))
print(tot_pathway.shape)
