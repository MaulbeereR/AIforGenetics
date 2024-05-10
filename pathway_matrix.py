import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

count_file_path = '../output/sorted_pathway_counts.txt'
pathway_file_path = '../output/pathways.txt'

sorted_pathways = np.loadtxt(count_file_path, dtype=str)
pathway_list = sorted_pathways[:, 0]

with open(pathway_file_path, 'r') as file:
    pathways = file.readlines()

pathways = [line.strip().split('|') for line in pathways]

binary_matrix = np.zeros((len(pathways), len(pathway_list)), dtype=int)

for i, pathway in enumerate(pathways):
    for j, pathway_name in enumerate(pathway_list):
        binary_matrix[i, j] = int(pathway_name in pathway)

df = pd.DataFrame(binary_matrix, columns=pathway_list)
print(df.head())
print("Number of samples:", len(pathways))
print("Number of pathways:", len(pathway_list))

# df.to_csv('../output/pathway_matrix.csv', index=False)

pathway_counts = df.sum(axis=0)
print(pathway_counts)
