import numpy as np
from scipy.interpolate import interp1d
from functools import partial
import h5py
from tqdm import tqdm

def split_hdf5_dataset_by_time(input_hdf5_path, output_hdf5_path, split_length_T):
    """
    Create a new HDF5 dataset by splitting each sample into two samples along the time dimension.
    
    Parameters:
    -----------
    input_hdf5_path : str or Path
        Path to the input HDF5 file
    output_hdf5_path : str or Path  
        Path to the output HDF5 file
    split_length_T : int
        Length of each split sample along time dimension
    """
    
    # Open input file
    with h5py.File(input_hdf5_path, 'r') as input_f:
        print("Reading input dataset structure...")
        
        # Get all sample dataset names
        time_series_group = input_f['time_series']
        input_dataset_names = sorted(time_series_group.keys())
        
        # Read metadata
        metadata_group = input_f['metadata']
        input_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in metadata_group['subjects'][:]]
        input_sessions = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in metadata_group['sessions'][:]]
        input_file_ids = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in metadata_group['file_ids'][:]]
        input_file_paths = [s.decode('utf-8') if isinstance(s, bytes) else s 
                            for s in metadata_group['file_paths'][:]]
        input_shapes = metadata_group['shapes'][:]
        
        print(f"Found {len(input_dataset_names)} samples in input dataset")
        
        # Validate that samples can be split
        valid_samples = []
        for i, dataset_name in enumerate(input_dataset_names):
            time_points = input_shapes[i][1]
            valid_samples.append(i)
        
        print(f"{len(valid_samples)} samples can be split (need >= {2*split_length_T} time points)")
        
        if not valid_samples:
            logger.error("No samples can be split with the given parameters")
            return False
        
        # Create output file
        with h5py.File(output_hdf5_path, 'w') as output_f:
            # Create groups
            output_time_series_group = output_f.create_group('time_series')
            output_metadata_group = output_f.create_group('metadata')
            
            # Prepare output metadata lists
            output_subjects = []
            output_sessions = []
            output_file_ids = []
            output_file_paths = []
            output_dataset_names = []
            output_shapes = []
            
            output_sample_idx = 0
            
            print("Processing samples...")
            for i in tqdm(valid_samples, desc="Splitting samples"):
                input_dataset_name = input_dataset_names[i]
                
                # Load the original time series data
                original_data = time_series_group[input_dataset_name][:]
                time_points, num_rois = original_data.shape
                
                # Split into two samples
                # First split: time points 0 to split_length_T-1
                split1_data = original_data[:, :split_length_T]
                # Second split: time points split_length_T to 2*split_length_T-1
                split2_data = original_data[:, split_length_T:2*split_length_T]
                
                splits = [split1_data, split2_data] if split2_data.shape[1] == split_length_T else [split1_data]

                # Save both splits
                for split_idx, split_data in enumerate(splits):
                    output_dataset_name = f"sample_{output_sample_idx:06d}"
                    
                    # Store time series data
                    output_time_series_group.create_dataset(
                        output_dataset_name,
                        data=split_data.astype(np.float32),
                        compression='gzip'
                    )
                    
                    # Store metadata (same for both splits)
                    output_subjects.append(input_subjects[i])
                    output_sessions.append(input_sessions[i])
                    # Modify file_id to indicate split
                    original_file_id = input_file_ids[i]
                    split_file_id = f"{original_file_id}_split{split_idx+1}"
                    output_file_ids.append(split_file_id)
                    output_file_paths.append(input_file_paths[i])
                    output_dataset_names.append(output_dataset_name)
                    output_shapes.append([split_length_T, num_rois])
                    
                    output_sample_idx += 1
            
            print(f"Created {len(output_subjects)} split samples")
            
            # Save metadata to output file
            print("Saving metadata...")
            
            # Convert strings to bytes for HDF5 storage
            subjects_bytes = [s.encode('utf-8') for s in output_subjects]
            sessions_bytes = [s.encode('utf-8') for s in output_sessions]
            file_ids_bytes = [s.encode('utf-8') for s in output_file_ids]
            file_paths_bytes = [s.encode('utf-8') for s in output_file_paths]
            dataset_names_bytes = [s.encode('utf-8') for s in output_dataset_names]
            
            # Create metadata datasets
            output_metadata_group.create_dataset('subjects', data=subjects_bytes, 
                                                dtype=h5py.string_dtype(encoding='utf-8'))
            output_metadata_group.create_dataset('sessions', data=sessions_bytes,
                                                dtype=h5py.string_dtype(encoding='utf-8'))
            output_metadata_group.create_dataset('file_ids', data=file_ids_bytes,
                                                dtype=h5py.string_dtype(encoding='utf-8'))
            output_metadata_group.create_dataset('file_paths', data=file_paths_bytes,
                                                dtype=h5py.string_dtype(encoding='utf-8'))
            output_metadata_group.create_dataset('dataset_names', data=dataset_names_bytes,
                                                dtype=h5py.string_dtype(encoding='utf-8'))
            output_metadata_group.create_dataset('shapes', data=np.array(output_shapes, dtype=np.int32))
            
    print(f"Successfully created split dataset: {output_hdf5_path}")
    print(f"Original samples: {len(input_dataset_names)}")
    print(f"Valid samples for splitting: {len(valid_samples)}")
    print(f"Output samples: {len(output_subjects)}")
    


def resample_fmri_linear(data, source_tr, target_tr=2, method='linear', step=3):
    """
    Resample 4D fMRI data from source TR to target TR using linear interpolation.
    
    Parameters:
    - data: 4D numpy array (..., time)
    - source_tr: original TR in seconds
    - target_tr: desired TR in seconds
    
    Returns:
    - resampled_data: 4D array with new temporal resolution
    """
    n_timepoints = data.shape[-1]
    duration = source_tr * (n_timepoints - 1)
    
    # Original and new time grids
    time_orig = np.linspace(0, duration, n_timepoints)

    if method:
        n_new_timepoints = int(np.round(duration / target_tr)) + 1
        time_new = np.linspace(0, duration, n_new_timepoints)
    else:
        n_new_timepoints = n_timepoints // 3

    original_shape = data.shape
    data_2d = data.reshape(-1, n_timepoints)

    resampled_2d = np.zeros((data_2d.shape[0], n_new_timepoints))
    
    if method:
        for i in range(data_2d.shape[0]):
            f = interp1d(time_orig, data_2d[i, :], 
                        kind=method, fill_value='extrapolate')
            resampled_2d[i, :] = f(time_new)
    else:
        for i in range(data_2d.shape[0]):
            resampled_2d[i, :] = data_2d[i, ::step][:n_new_timepoints]

    new_shape = original_shape[:-1] + (n_new_timepoints,)
    return resampled_2d.reshape(new_shape)


def apply_extra_processing_to_hdf5_batched(input_hdf5_path, output_hdf5_path, extra_processing_func, batch_size=100):
    """
    Process datasets in batches and save to a new HDF5 file to save memory.
    
    Parameters:
    - input_hdf5_path: path to input HDF5 file
    - output_hdf5_path: path to output HDF5 file (will be created)
    - extra_processing_func: function to apply to each dataset
    - batch_size: number of datasets to process in each batch
    """
    with h5py.File(input_hdf5_path, 'r') as f_in, h5py.File(output_hdf5_path, 'w') as f_out:
        if 'time_series' not in f_in:
            raise RuntimeError('No time_series group found in the input HDF5 file.')
        
        input_time_series_group = f_in['time_series']
        output_time_series_group = f_out.create_group('time_series')
        output_meta_group = f_out.create_group('metadata')
        
        # Copy other groups/datasets that might exist (except time_series and metadata)
        for key in f_in.keys():
            if key not in ['time_series', 'metadata']:
                f_in.copy(key, f_out)
        
        # Copy existing metadata (except shapes which will be updated)
        if 'metadata' in f_in:
            for key in f_in['metadata'].keys():
                if key != 'shapes':
                    f_in['metadata'].copy(key, output_meta_group)
        
        dataset_names = list(input_time_series_group.keys())
        total_datasets = len(dataset_names)
        updated_shapes = []
        
        print(f"Processing {total_datasets} datasets in batches of {batch_size}...")
        print(f"Input file: {input_hdf5_path}")
        print(f"Output file: {output_hdf5_path}")
        
        # Process in batches
        for batch_start in tqdm(range(0, total_datasets, batch_size)):
            batch_end = min(batch_start + batch_size, total_datasets)
            batch_names = dataset_names[batch_start:batch_end]
            
            batch_shapes = []
            for dname in batch_names:
                original_data = input_time_series_group[dname][()].T
                processed_data = extra_processing_func(original_data)
                
                output_time_series_group.create_dataset(
                    dname, 
                    data=processed_data.astype(np.float32), 
                    compression='gzip'
                )
                
                batch_shapes.append(processed_data.shape)
            
            updated_shapes.extend(batch_shapes)
            print(f"Processed batch {batch_start//batch_size + 1}/{(total_datasets-1)//batch_size + 1}")
        
        # Create updated shapes metadata
        shapes_np = np.array(updated_shapes, dtype=np.int32)
        output_meta_group.create_dataset('shapes', data=shapes_np, maxshape=(None, 2))
        
        print(f"Successfully processed all {total_datasets} datasets!")
        print(f"New file saved to: {output_hdf5_path}")
    
    return True

if __name__ == '__main__':
    # func = partial(resample_fmri_linear, target_tr=2, source_tr=0.735, method='', step=3)

    # apply_extra_processing_to_hdf5_batched(
    #     'data/HCP/fmri/TianS3/data.h5', 
    #     'data/HCP/fmri/TianS3/data_resampled_step.h5',  # New output file
    #     func
    # )

    split_hdf5_dataset_by_time('data/HCP/fmri/TianS3/data_resampled.h5', 
                              'data/HCP/fmri/TianS3/data_resampled_split180*2.h5', 
                              split_length_T=180)