import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count


_ATLAS_IMG = None

def _init_worker(atlas_path):
    global _ATLAS_IMG
    _ATLAS_IMG = nib.load(atlas_path)

def _process_dataset(args):
    key, data = args
    if data.ndim == 4:
        processed = gather_voxel_within_parcel(data, _ATLAS_IMG, 259)
    else:
        # processed = data[:, :140]
        processed = data.T
    # return key, processed.transpose(1, 0, 2)
    return key, processed

def postprocess_h5(h5_path, atlas_path='data/atlas/TianS3_mni.nii.gz', num_workers=None):
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)
    with h5py.File(h5_path, 'r+') as f:
        keys = list(f['time_series'].keys())
        task_iter = ((key, f['time_series'][key][:]) for key in keys)
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(atlas_path,)) as pool:
            iterator = pool.imap_unordered(_process_dataset, task_iter)
            for key, processed in tqdm(iterator, total=len(keys), desc='Postprocessing'):
                del f['time_series'][key]
                f['time_series'].create_dataset(key, data=processed)

if __name__ == "__main__":
    h5_path = 'data/EHBS/fmri/TianS3/data_resampled.h5'
    atlas_img_path = 'data/atlas/TianS3_mni.nii.gz'
    postprocess_h5(h5_path, atlas_img_path)