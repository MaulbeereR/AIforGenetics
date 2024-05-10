import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('../output/sorted_pathway_counts.txt', header=None, sep="\t", names=["pathway", "count"])

df['levels'] = df['pathway'].apply(lambda x: x.split('_'))

df['main_category'] = df['levels'].apply(lambda x: x[0])

print('main_category:')
print(df['main_category'].value_counts())

df['sub_category'] = df['levels'].apply(lambda x: x[1] if len(x) > 1 else None)
print('sub_category:')
print(df['sub_category'].value_counts())
print(df['sub_category'].unique().shape)

