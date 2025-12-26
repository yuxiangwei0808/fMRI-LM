import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import logging

def remove_samples_by_indices(input_h5_path, output_h5_path, indices_to_remove):
    """
    Remove samples from HDF5 dataset by their indices.
    
    Args:
        input_h5_path: Path to input HDF5 file
        output_h5_path: Path for output HDF5 file
        indices_to_remove: List or set of sample indices to remove
    """
    indices_to_remove = set(indices_to_remove)
    
    with h5py.File(input_h5_path, 'r') as input_f:
        # Get total number of samples from metadata
        total_samples = len(input_f['metadata/subjects'])
        print(f"Total samples in dataset: {total_samples}")
        
        # Determine which samples to keep
        indices_to_keep = [i for i in range(total_samples) if i not in indices_to_remove]
        print(f"Samples to keep: {len(indices_to_keep)}")
        print(f"Samples to remove: {len(indices_to_remove)}")
        
        # Create new HDF5 file
        with h5py.File(output_h5_path, 'w') as output_f:
            # Create groups
            output_f.create_group("time_series")
            meta_group = output_f.create_group("metadata")
            
            # Copy metadata for kept samples
            kept_subjects = []
            kept_sessions = []
            kept_file_ids = []
            kept_file_paths = []
            kept_dataset_names = []
            kept_shapes = []
            
            for new_idx, old_idx in enumerate(indices_to_keep):
                # Read metadata from old file
                subject = input_f['metadata/subjects'][old_idx].decode('utf-8')
                session = input_f['metadata/sessions'][old_idx].decode('utf-8')
                file_id = input_f['metadata/file_ids'][old_idx].decode('utf-8')
                file_path = input_f['metadata/file_paths'][old_idx].decode('utf-8')
                old_dataset_name = input_f['metadata/dataset_names'][old_idx].decode('utf-8')
                # shape = input_f['metadata/shapes'][old_idx]
                
                # Create new dataset name
                new_dataset_name = f"sample_{new_idx:06d}"
                
                # Copy time series data with new name
                old_data = input_f[f'time_series/{old_dataset_name}'][:]
                output_f.create_dataset(
                    f"time_series/{new_dataset_name}",
                    data=old_data,
                    compression='lzf'
                )
                
                # Collect metadata
                kept_subjects.append(subject)
                kept_sessions.append(session)
                kept_file_ids.append(file_id)
                kept_file_paths.append(file_path)
                kept_dataset_names.append(new_dataset_name)
                # kept_shapes.append(shape)
            
            # Save metadata to new file
            meta_group.create_dataset('subjects', data=[s.encode('utf-8') for s in kept_subjects],
                                    dtype=h5py.string_dtype(encoding='utf-8'))
            meta_group.create_dataset('sessions', data=[s.encode('utf-8') for s in kept_sessions],
                                    dtype=h5py.string_dtype(encoding='utf-8'))
            meta_group.create_dataset('file_ids', data=[s.encode('utf-8') for s in kept_file_ids],
                                    dtype=h5py.string_dtype(encoding='utf-8'))
            meta_group.create_dataset('file_paths', data=[s.encode('utf-8') for s in kept_file_paths],
                                    dtype=h5py.string_dtype(encoding='utf-8'))
            meta_group.create_dataset('dataset_names', data=[s.encode('utf-8') for s in kept_dataset_names],
                                    dtype=h5py.string_dtype(encoding='utf-8'))
            # meta_group.create_dataset('shapes', data=np.array(kept_shapes), dtype=np.int32)
    
    print(f"Successfully created filtered dataset: {output_h5_path}")


def remove_samples_by_subjects(input_h5_path, output_h5_path, subjects_to_remove):
    """
    Remove samples from HDF5 dataset by subject names.
    
    Args:
        input_h5_path: Path to input HDF5 file
        output_h5_path: Path for output HDF5 file
        subjects_to_remove: List or set of subject names to remove
    """
    subjects_to_remove = set(subjects_to_remove)
    
    with h5py.File(input_h5_path, 'r') as input_f:
        # Get all subjects and find indices to remove
        all_subjects = [int(s.decode('utf-8')) for s in input_f['metadata/subjects'][:]]
        indices_to_remove = [i for i, subj in enumerate(all_subjects) if subj in subjects_to_remove]
        
        print(f"Found {len(indices_to_remove)} samples to remove from {len(subjects_to_remove)} subjects")
        
        # Use the index-based removal method
        remove_samples_by_indices(input_h5_path, output_h5_path, indices_to_remove)


def remove_samples_by_shape(input_h5_path, output_h5_path, min_last_dim=149):
    """
    Remove samples from HDF5 dataset if their last dimension is less than a threshold.
    
    Args:
        input_h5_path: Path to input HDF5 file
        output_h5_path: Path for output HDF5 file
        min_last_dim: Minimum value for the last dimension (default: 149)
                      Samples with shape[-1] < min_last_dim will be removed
    """
    with h5py.File(input_h5_path, 'r') as input_f:
        # Get all shapes from metadata
        all_shapes = input_f['metadata/shapes'][:]
        total_samples = len(all_shapes)
        
        # Find indices of samples where shape[-1] < min_last_dim
        indices_to_remove = [i for i in range(total_samples) if all_shapes[i][-1] < min_last_dim]
        
        print(f"Total samples in dataset: {total_samples}")
        print(f"Samples with shape[-1] < {min_last_dim}: {len(indices_to_remove)}")
        
        if indices_to_remove:
            # Show some examples of shapes being removed
            print(f"Example shapes being removed: {[all_shapes[i].tolist() for i in indices_to_remove[:5]]}")
        
        # Use the index-based removal method
        remove_samples_by_indices(input_h5_path, output_h5_path, indices_to_remove)

    return indices_to_remove


if __name__ == "__main__":
    # excluded_subject = [1352930, 5967843, 5421032, 3987401, 5867340, 4093425, 1061622, 5359415, 4628315, 5781852, 1053118, 4929119]
    # remove_samples_by_subjects('data/UKB/fmri/TianS3/data_resampled.h5', 'data/UKB/fmri/TianS3/data_filtered.h5', excluded_subject)

    # indices_to_remove = remove_samples_by_shape('data/ABCD/fmri/TianS3/data_resampled.h5', 'data/ABCD/fmri/TianS3/data_filtered.h5', min_last_dim=149)
    # np.save('data/ABCD/fmri/TianS3/removed_sample_indices.npy', np.array(indices_to_remove))

    # remove_samples_by_indices('data/ABCD/fmri/fc/data_fc.h5', 'data/ABCD/fmri/fc/data_filtered.h5', np.load('data/ABCD/fmri/TianS3/removed_sample_indices.npy'))

        
    # df_fc = pd.read_csv('data/ABCD/fmri/descriptors_rewritten/fc_descriptors.csv')
    # df_subj_sess = [(df_fc['subject_id'][i], df_fc['session_id'][i]) for i in range(len(df_fc))]

    # missing_index = []
    # with h5py.File('data/ABCD/fmri/TianS3/data_resampled.h5', 'r') as f:
    #     # find the index that are not in df_fc
    #     for i in range(len(f['time_series'])):
    #         subj_id = f['metadata/subjects'][i].decode('utf-8')
    #         sess_id = f['metadata/sessions'][i].decode('utf-8')
    #         if (subj_id, sess_id) not in df_subj_sess:
    #             missing_index.append(i)
    missing_index = [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369]
    remove_samples_by_indices('data/ADHD200/fmri/TianS3/data_resampled.h5', 'data/ADHD200/fmri/TianS3/data_filtered.h5', missing_index)