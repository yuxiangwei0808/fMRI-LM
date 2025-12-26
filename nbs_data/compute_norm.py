import h5py
import numpy as np
from tqdm import tqdm

dataset_name = 'EHBS'
path = f'data/{dataset_name}/fmri/TianS3/data_resampled.h5'
file_handle = h5py.File(path, 'r')
data = file_handle['time_series']
keys = list(data.keys())

# keys = keys[:int(len(keys) * 0.8)]

all_data_mean = []
for k in tqdm(keys):
    # all_data_mean.append(data[k][:].mean(1))  # mean over time

    all_data_mean.append(data[k][:].mean(-1))  # mean over time

all_data_mean = np.stack(all_data_mean, 0)
assert all_data_mean.shape[1] == 450
medians_roi = np.median(all_data_mean, axis=0)
iqrs_roi = np.percentile(all_data_mean, 75, axis=0) - np.percentile(all_data_mean, 25, axis=0)
mean_roi, std_roi = np.mean(all_data_mean, axis=0), np.std(all_data_mean, axis=0)

if all_data_mean.ndim == 2: # original shape of each sample is (V T)
    medians, iqrs, mean, std = np.median(all_data_mean), np.percentile(all_data_mean, 75) - np.percentile(all_data_mean, 25), np.mean(all_data_mean), np.std(all_data_mean)
else:
    medians = np.median(all_data_mean, axis=(0, 1), keepdims=True)
    iqrs = np.percentile(all_data_mean, 75, axis=(0, 1), keepdims=True) - np.percentile(all_data_mean, 25, axis=(0, 1), keepdims=True)
    mean = np.mean(all_data_mean, axis=(0, 1), keepdims=True)
    std = np.std(all_data_mean, axis=(0, 1), keepdims=True)

# mask out -1
# all_data_mean = all_data_mean[all_data_mean != -1]
# medians, iqrs, mean, std = np.median(all_data_mean), np.percentile(all_data_mean, 75) - np.percentile(all_data_mean, 25), np.mean(all_data_mean), np.std(all_data_mean)

np.savez(f'data/{dataset_name}/fmri/TianS3/normalization_params.npz', medians_roi=medians_roi, iqrs_roi=iqrs_roi, mean_roi=mean_roi, std_roi=std_roi, 
         medians=medians, iqrs=iqrs, mean=mean, std=std)