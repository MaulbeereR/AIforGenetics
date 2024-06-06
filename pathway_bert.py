import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

data = pd.read_csv('../output/sorted_pathway_counts.txt', sep='\t', header=None)
pathways = data.iloc[:, 0].values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


features = []
batch_size = 32
for i in tqdm(range(0, len(pathways), batch_size), desc="Extracting BERT embeddings"):
    batch = list(pathways[i:i+batch_size])
    embeddings = get_bert_embeddings(batch)
    features.append(embeddings)
features = torch.cat(features, dim=0).numpy()


num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
labels = kmeans.labels_


clustered_pathways = {i: [] for i in range(num_clusters)}
for pathway, label in zip(pathways, labels):
    clustered_pathways[label].append(pathway)

with open('../output/bert_clustered_pathways.txt', 'w') as f:
    for cluster, pathways in clustered_pathways.items():
        f.write(f'Cluster {cluster}:\n')
        for pathway in pathways:
            f.write(f'  {pathway}\n')
        f.write('\n')
