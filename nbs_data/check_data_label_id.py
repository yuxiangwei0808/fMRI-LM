"""
Find the indices of subjects in the HDF5 file that have corresponding labels in the CSV metadata file.
Saves the valid indices to a NumPy file.
"""

import h5py
import polars as pl
import numpy as np

def check_row(row, target_name):
    return row[0, target_name] is not None and row[0, target_name] != '' and not np.isnan(row[0, target_name])

def get_label_indices(dataset_name, target_name=None):
    df = pl.read_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical_gpt.csv')
    subj_sess = [(str(row['subject_id']), str(row['session_id'])) for row in df.iter_rows(named=True)]
    valid_inds = []
    df = df.with_columns(pl.col('subject_id').cast(pl.String))

    with h5py.File(f'data/{dataset_name}/fmri/TianS3/data_resampled.h5', 'r') as f:
        for i in range(len(f['time_series'])):
            subj_id = f['metadata/subjects'][i].decode('utf-8')
            sess_id = f['metadata/sessions'][i].decode('utf-8')
            if (subj_id, sess_id) in subj_sess:
                if target_name is not None:
                    row = df.filter((pl.col('subject_id') == subj_id) & (pl.col('session_id') == sess_id))
                    if isinstance(target_name, list):
                        if all(check_row(row, tn) for tn in target_name):
                            valid_inds.append(i)
                    else:
                        if check_row(row, target_name):
                            valid_inds.append(i)
                elif target_name is None:
                    valid_inds.append(i)
        print(f'Found {len(valid_inds)} subjects with labels out of {len(f["time_series"])} total subjects.')
    if target_name is None:
        np.save(f'data/{dataset_name}/fmri/inds_with_label.npy', valid_inds)
    else:
        if isinstance(target_name, list):
            target_name = 'mq'
            # target_name = 'open'
        np.save(f'data/{dataset_name}/fmri/inds_with_label_{target_name}.npy', valid_inds)

def get_desc_indices():
    dataset_name = 'UKB'
    df = pl.read_csv(f'data/{dataset_name}/fmri/descriptors_rewritten/ica_descriptors.csv')
    subj_sess = [(str(row['subject_id']), str(row['session_id'])) for row in df.iter_rows(named=True)]
    valid_inds = []

    with h5py.File(f'data/{dataset_name}/fmri/TianS3/data_resampled.h5', 'r') as f:
        for i in range(len(f['time_series'])):
            subj_id = f['metadata/subjects'][i].decode('utf-8')
            sess_id = f['metadata/sessions'][i].decode('utf-8')
            if (subj_id, sess_id) in subj_sess:
                valid_inds.append(i)
        print(f'Found {len(valid_inds)} subjects with descriptors out of {len(f["time_series"])} total subjects.')
    np.save(f'data/{dataset_name}/fmri/inds_with_desc.npy', valid_inds)

if __name__ == "__main__":
    get_label_indices('ADNI', ['AD', 'apoe4_enc'])