import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

file_path = '../filtered_FCS_data_2024-03-07_V2.csv'
df = pd.read_csv(file_path)

pathways_data = df['pathways']

# Split the pathways content and create a set of unique pathways
unique_pathways = set()
for pathways in pathways_data:
    unique_pathways.update(pathways.split('|'))

unique_pathways_count = len(unique_pathways)
print(unique_pathways_count)


pathway_counts = Counter()
for pathways in pathways_data:
    pathway_counts.update(pathways.split('|'))

plt.figure(figsize=(10, 6))
plt.hist(pathway_counts.values(), bins=range(1, max(pathway_counts.values()) + 1), align='left')
plt.xlabel('Number of Occurrences')
plt.ylabel('Number of Pathways')
plt.title('Histogram of Pathway Occurrences')
plt.show()

sorted_pathway_counts = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)

output_file_path = '/mnt/data/sorted_pathway_counts.txt'
with open(output_file_path, 'w') as f:
    for pathway, count in sorted_pathway_counts:
        f.write(f'{pathway}\t{count}\n')

average_unique_pathways_per_row = sum(len(set(pathways.split('|'))) for pathways in pathways_data) / len(pathways_data)
print(average_unique_pathways_per_row)
