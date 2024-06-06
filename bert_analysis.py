import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

with open('../output/bert/bert_clustered_pathways.txt', 'r') as file:
    lines = file.readlines()


clustered_pathways = {}
current_cluster = None
for line in lines:
    if line.startswith('Cluster'):
        current_cluster = int(line.split()[1][:-1])
        clustered_pathways[current_cluster] = []
    elif line.strip():
        clustered_pathways[current_cluster].append(line.strip())


def get_function_type(pathway):
    return pathway.split('_')[1]


cluster_function_distribution = {}
for cluster, pathways in clustered_pathways.items():
    function_types = [get_function_type(pathway) for pathway in pathways]
    function_count = Counter(function_types)
    cluster_function_distribution[cluster] = function_count


def plot_cluster_distribution(cluster_function_distribution):
    for cluster, function_count in cluster_function_distribution.items():
        labels, values = zip(*function_count.items())
        plt.figure(figsize=(20, 10))
        plt.bar(labels, values)
        plt.title(f'Cluster {cluster} Type Distribution')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'../output/bert/cluster_{cluster}_distribution.png')


plot_cluster_distribution(cluster_function_distribution)


for cluster, function_count in cluster_function_distribution.items():
    total = sum(function_count.values())
    print(f'Cluster {cluster}:')
    for function, count in function_count.items():
        print(f'  {function}: {count} ({count / total:.2%})')
    print()
