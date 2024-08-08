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

unique_pathways = ['GOBP_REGULATION_OF_IMMUNE_SYSTEM_PROCESS','GOBP_DEFENSE_RESPONSE','GOBP_CELL_ACTIVATION','GOBP_BIOLOGICAL_PROCESS_INVOLVED_IN_INTERSPECIES_INTERACTION_BETWEEN_ORGANISMS','GOBP_POSITIVE_REGULATION_OF_IMMUNE_SYSTEM_PROCESS','GOBP_LYMPHOCYTE_ACTIVATION','GOBP_DEFENSE_RESPONSE_TO_OTHER_ORGANISM','GOBP_T_CELL_ACTIVATION','GOBP_CELL_ADHESION','GOBP_REGULATION_OF_CELL_ADHESION','GOBP_POSITIVE_REGULATION_OF_CELL_ADHESION','GOBP_HOMEOSTATIC_PROCESS','GOBP_POSITIVE_REGULATION_OF_CELL_POPULATION_PROLIFERATION','GOBP_REGULATION_OF_CELL_ACTIVATION','GOBP_POSITIVE_REGULATION_OF_MULTICELLULAR_ORGANISMAL_PROCESS','GOBP_POSITIVE_REGULATION_OF_GENE_EXPRESSION','GOBP_REGULATION_OF_LYMPHOCYTE_ACTIVATION','GOBP_CELL_CELL_ADHESION','GOBP_CYTOKINE_PRODUCTION','GOBP_REGULATION_OF_CELL_CELL_ADHESION','GOBP_LEUKOCYTE_PROLIFERATION','GOBP_LEUKOCYTE_CELL_CELL_ADHESION','GOBP_REGULATION_OF_T_CELL_ACTIVATION','GOBP_REGULATION_OF_CATALYTIC_ACTIVITY','GOBP_VESICLE_MEDIATED_TRANSPORT','GOBP_CELLULAR_CATABOLIC_PROCESS','GOBP_HEMOPOIESIS','GOBP_T_CELL_PROLIFERATION','GOCC_ENDOPLASMIC_RETICULUM','GOBP_REGULATION_OF_RESPONSE_TO_STRESS','GOBP_REGULATION_OF_CELL_DIFFERENTIATION','GOBP_INTRACELLULAR_TRANSPORT','GOBP_REGULATION_OF_LEUKOCYTE_PROLIFERATION','GOBP_POSITIVE_REGULATION_OF_CELL_ACTIVATION','GOCC_GOLGI_APPARATUS','GOMF_HYDROLASE_ACTIVITY_ACTING_ON_ESTER_BONDS','GOBP_REGULATION_OF_IMMUNE_RESPONSE','GOBP_TRANSMEMBRANE_TRANSPORT','GOBP_REGULATION_OF_T_CELL_PROLIFERATION','GOBP_SMALL_MOLECULE_METABOLIC_PROCESS','GOBP_CYTOSKELETON_ORGANIZATION','GOCC_MICROTUBULE_CYTOSKELETON','GOBP_POSITIVE_REGULATION_OF_IMMUNE_RESPONSE','GOBP_REGULATION_OF_RESPONSE_TO_EXTERNAL_STIMULUS','GOBP_REGULATION_OF_DEFENSE_RESPONSE','GOBP_CHEMICAL_HOMEOSTASIS','GOBP_POSITIVE_REGULATION_OF_LYMPHOCYTE_ACTIVATION','GOMF_CYTOSKELETAL_PROTEIN_BINDING','GOBP_LEUKOCYTE_DIFFERENTIATION','GOBP_POSITIVE_REGULATION_OF_LEUKOCYTE_CELL_CELL_ADHESION','GOBP_POSITIVE_REGULATION_OF_CELL_CELL_ADHESION','GOBP_CELLULAR_HOMEOSTASIS','GOBP_INNATE_IMMUNE_RESPONSE','GOMF_MOLECULAR_FUNCTION_ACTIVATOR_ACTIVITY','GOBP_MAINTENANCE_OF_LOCATION','GOBP_MONOATOMIC_ION_HOMEOSTASIS','GOBP_LIPID_METABOLIC_PROCESS','GOCC_MICROTUBULE_ORGANIZING_CENTER','GOBP_PHOSPHORYLATION','GOBP_MONOATOMIC_ION_TRANSPORT','MP_DECREASED_TUMOR_NECROSIS_FACTOR_SECRETION','GOBP_POSITIVE_REGULATION_OF_CYTOKINE_PRODUCTION','GOBP_MONONUCLEAR_CELL_DIFFERENTIATION','GOBP_MONOATOMIC_CATION_TRANSPORT','GOBP_MONOATOMIC_ION_TRANSMEMBRANE_TRANSPORT','GOMF_PHOSPHORIC_ESTER_HYDROLASE_ACTIVITY','GOMF_TUBULIN_BINDING','GOBP_INORGANIC_ION_TRANSMEMBRANE_TRANSPORT','GOBP_CALCIUM_ION_TRANSPORT','GOBP_G_PROTEIN_COUPLED_RECEPTOR_SIGNALING_PATHWAY','GOBP_CALCIUM_ION_TRANSMEMBRANE_TRANSPORT','GOBP_CALCIUM_ION_TRANSMEMBRANE_IMPORT_INTO_CYTOSOL','GOMF_PURINE_NUCLEOTIDE_BINDING','GOBP_REGULATION_OF_INTRACELLULAR_SIGNAL_TRANSDUCTION','GOBP_POSITIVE_REGULATION_OF_MOLECULAR_FUNCTION','GOBP_MAINTENANCE_OF_LOCATION_IN_CELL','GOBP_INORGANIC_ION_HOMEOSTASIS','GOCC_CENTROSOME','GOBP_POSITIVE_REGULATION_OF_SIGNAL_TRANSDUCTION','GOBP_CALCIUM_ION_HOMEOSTASIS','GOBP_SEQUESTERING_OF_CALCIUM_ION','GOBP_REGULATION_OF_MULTICELLULAR_ORGANISMAL_DEVELOPMENT','GOBP_REGULATION_OF_PROTEIN_MODIFICATION_PROCESS','GOBP_REGULATION_OF_PHOSPHORUS_METABOLIC_PROCESS','GOBP_CELLULAR_LIPID_METABOLIC_PROCESS','GOBP_MICROTUBULE_BASED_PROCESS','GOBP_REGULATION_OF_CELL_DEVELOPMENT','GOBP_IMMUNE_EFFECTOR_PROCESS','GOBP_PEPTIDYL_AMINO_ACID_MODIFICATION','GOBP_POSITIVE_REGULATION_OF_LEUKOCYTE_PROLIFERATION','GOBP_ORGANOPHOSPHATE_METABOLIC_PROCESS','GOMF_ADENYL_NUCLEOTIDE_BINDING','GOCC_SYNAPSE','GOBP_CELL_CYCLE','GOBP_MICROTUBULE_CYTOSKELETON_ORGANIZATION','GOBP_SUPRAMOLECULAR_FIBER_ORGANIZATION','GOBP_CELLULAR_RESPONSE_TO_STRESS','GOMF_ENZYME_REGULATOR_ACTIVITY','GOBP_CELL_MOTILITY']
# print(unique_pathways.__len__())
input_path = '../'
metadata = pd.read_csv(input_path + 'filtered_FCS_data_2024-03-07_V2.csv')


def process_raw_data(cell_count, input_path, metadata, sample_count):
    input_path += 'fcs_data_V2/'

    input_filelist = os.listdir(input_path)

    all_data_list = []
    all_labels_list = []

    for f_idx, fcs_filename in enumerate(input_filelist):

        if f_idx <= sample_count:
            pathways = metadata.loc[f_idx, 'pathways']
            print(f_idx, fcs_filename, pathways)

            label = [0] * 99
            pathways_list = pathways.split('|')
            for pw in pathways_list:
                if pw in unique_pathways:
                    label[unique_pathways.index(pw)] = 1
            print(label)

            cell_data = common.read_FCS(input_path + fcs_filename)
            print(cell_data.shape)

            if cell_data.shape[0] > cell_count:
                trimmed_data = cell_data[0:cell_count, :]
                if trimmed_data.shape == (cell_count, 6):
                    all_data_list.append(trimmed_data)
                    all_labels_list.append(label)

    all_data = np.stack(all_data_list)
    all_labels = np.stack(all_labels_list)

    return all_data, all_labels


data_2D_filename = '../output/dataset/new_data.npy'
label_filename = '../output/dataset/new_labels.npy'

num_classes = 2
cell_count = 30000
ch_count = 6

all_data, all_labels = process_raw_data(cell_count=cell_count, input_path=input_path, metadata=metadata,
                                        sample_count=10000)
np.save(data_2D_filename, all_data)
np.save(label_filename, all_labels)

data = np.load('../output/dataset/new_data.npy', allow_pickle=True)
labels = np.load('../output/dataset/new_labels.npy', allow_pickle=True)

print('data.shape: ', data.shape)
print('labels.shape: ', labels.shape)

data_old = np.load('../output/dataset/all_data_V4.npy', allow_pickle=True)
print('data_old.shape: ', data_old.shape)

print(np.array_equal(data, data_old))

