import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

matrix_file_path = '../output/pathway_matrix.csv'
df = pd.read_csv(matrix_file_path)

clusters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for n_clusters in clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=24)
    cluster = kmeans.fit_predict(df)

    tsne = TSNE(n_components=2, random_state=24)
    tsne_results = tsne.fit_transform(df)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{n_clusters} Clusters')
    plt.show()

