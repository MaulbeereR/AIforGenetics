import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

matrix_file_path = '../output/pathway_matrix.csv'
df = pd.read_csv(matrix_file_path)
sample_names = df.iloc[:, 0].values
df_data = df.iloc[:, 1:]
df_transposed = df_data.transpose()
column_names = df_data.columns

clusters = [10, 20]
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', '#800080', '#008080',
          '#C0C0C0', '#FFA500', '#A52A2A', '#8A2BE2', '#DEB887', '#5F9EA0', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC']

for n_clusters in clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=24)
    cluster = kmeans.fit_predict(df_transposed)

    tsne = TSNE(n_components=2, random_state=24)
    tsne_results = tsne.fit_transform(df_transposed)

    plt.figure(figsize=(12, 8))
    cmap = ListedColormap(colors[:n_clusters])
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster, cmap=cmap, alpha=0.6)
    plt.colorbar(scatter, ticks=np.arange(n_clusters))
    plt.title(f'{n_clusters} Clusters')
    plt.show()

    output_df = pd.DataFrame({
        'Sample': column_names,
        'Cluster': cluster
    })
    output_df.to_csv(f'../output/kmeans_pathway/{n_clusters}_cluster.csv', index=False)

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

