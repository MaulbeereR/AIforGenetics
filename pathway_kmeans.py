import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

matrix_file_path = '../output/pathway_matrix.csv'
df = pd.read_csv(matrix_file_path)
sample_names = df.iloc[:, 0].values
df_data = df.iloc[:, 1:]
df_transposed = df_data.transpose()
column_names = df_data.columns

clusters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for n_clusters in clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=24)
    cluster = kmeans.fit_predict(df_transposed)

    tsne = TSNE(n_components=2, random_state=24)
    tsne_results = tsne.fit_transform(df_transposed)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{n_clusters} Clusters')
    plt.show()

    # cluster_dict = {}
    # for i in range(n_clusters):
    #     cluster_dict[i] = column_names[cluster == i].tolist()
    #
    # output_file = f'../cluster_results_{n_clusters}.csv'
    # with open(output_file, 'w') as f:
    #     for key, values in cluster_dict.items():
    #         f.write(f'Cluster {key},')
    #         f.write(','.join(values))
    #         f.write('\n')

