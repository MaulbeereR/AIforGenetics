import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

data_folder = f'../output/tensor_data/'

file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]

all_data = []
for file_path in file_paths:
    print(file_path)
    data = np.load(file_path)
    all_data.append(data)

all_data = np.stack(all_data).flatten()

mean = np.mean(all_data)
std = np.std(all_data)
min_val = np.min(all_data)
max_val = np.max(all_data)
print(mean, std, min_val, max_val)


all_data_reshaped = all_data.reshape(-1, 1)
scaler = preprocessing.StandardScaler().fit(all_data_reshaped)
scaled_data = scaler.transform(all_data_reshaped)
scaled_data = scaled_data.flatten()
print(np.mean(scaled_data), np.std(scaled_data), np.min(scaled_data), np.max(scaled_data))

print(scaled_data.shape)
reshaped_data = scaled_data.reshape(len(file_paths), 15, 80, 80)

for i, file_path in enumerate(file_paths):
    output_path = os.path.join(f'../output/tensor_data_norm/', os.path.basename(file_path))
    np.save(output_path, reshaped_data[i])
