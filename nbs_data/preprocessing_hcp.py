import os
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img, mean_img
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging
import json
from pathlib import Path
import h5py
import time
import argparse

from temporal_resample import resample_fmri_linear


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HCP fMRI preprocessing with multiprocessing")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of processes to use (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for saving data (default: 50)")
    parser.add_argument("--consolidate", action="store_true",
                       help="Only create consolidated numpy array from existing HDF5 data")
    parser.add_argument("--root-dir", type=str, 
                       default="/data/qneuromark/Data/HCP/Data_BIDS/Preprocess_Data/",
                       help="Root directory of the dataset")
    parser.add_argument("--output-dir", type=str, default="./hcp_preprocessing_output",
                       help="Output directory for processed data")
    parser.add_argument('--atlas_name', type=str, default='TianS3',
                       help='Atlas name to use for preprocessing (default: Schaefer)')
    parser.add_argument('--gather_parcel_voxel', action='store_true', default=False, 
                        help='instead of averaging within parcel, gather all voxel time series within each parcel')
    return parser.parse_args()

# Global variables (will be updated by command line arguments)
OUTPUT_DIR = None
BATCH_SIZE = 50
N_PROCESSES = None
root_dir = None
PROGRESS_FILE = None
ERROR_LOG_FILE = None
HDF5_FILE = None
METADATA_FILE = None
MAX_VOXELS = None  # only used if gather_parcel_voxel is True
logger = None

def main():
    args = parse_arguments()
    
    # Update global configuration based on arguments
    global OUTPUT_DIR, BATCH_SIZE, N_PROCESSES, root_dir
    global PROGRESS_FILE, ERROR_LOG_FILE, HDF5_FILE, METADATA_FILE, logger, MAX_VOXELS
    
    OUTPUT_DIR = Path(args.output_dir)
    BATCH_SIZE = args.batch_size
    N_PROCESSES = args.processes if args.processes else min(8, mp.cpu_count())
    root_dir = args.root_dir
    
    # Update file paths
    PROGRESS_FILE = OUTPUT_DIR / "progress.json"
    ERROR_LOG_FILE = OUTPUT_DIR / "errors.log"
    HDF5_FILE = OUTPUT_DIR / f"data_resampled_raw.h5"
    METADATA_FILE = OUTPUT_DIR / "HCP_fMRI_metadata.csv"
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(ERROR_LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    if args.consolidate:
        create_final_numpy_array(args)
        return
    
    # Run main preprocessing
    run_preprocessing(args)


def load_progress():
    """Load processing progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
        # Handle old format migration
        if "processed_files" not in progress:
            progress["processed_files"] = []
        return progress
    return {"processed_files": [], "total_processed": 0, "last_batch_processed": [], "last_batch_idx": -1}

def save_progress(progress):
    """Save processing progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_file_id(file_path, subject, session):
    """Generate unique identifier for a file."""
    return f"{subject}_{session}_{Path(file_path).name}"

def gather_voxel_within_parcel(fmri_img, atlas_img, max_voxels):
    """Gather voxel time series within each parcel up to max_voxels."""
    fmri_data = fmri_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    
    labels = np.unique(atlas_data)
    labels = labels[labels != 0]  # Exclude background
    
    all_time_series = []
    
    for label in labels:
        mask = atlas_data == label
        voxel_time_series = fmri_data[mask]
        
        if voxel_time_series.shape[0] > max_voxels:
            # Randomly sample max_voxels if more than max_voxels
            indices = np.random.choice(voxel_time_series.shape[0], max_voxels, replace=False)
            voxel_time_series = voxel_time_series[indices]
        elif voxel_time_series.shape[0] < max_voxels:
            # Pad with -1 if fewer than max_voxels
            padding = np.full((max_voxels - voxel_time_series.shape[0], voxel_time_series.shape[1]), -1)
            voxel_time_series = np.vstack([voxel_time_series, padding])
        
        all_time_series.append(voxel_time_series)
    
    return np.array(all_time_series).transpose(-1, -2)  # Shape: (n_parcels, n_timepoints, max_voxels)

def process_single_file(args):
    """Process a single fMRI file and return time series data."""
    file_path, subject, session, atlas_img_path, max_voxels = args

    # Load and process fMRI file
    fmri_img = nib.load(file_path)
    
    # Load atlas (each process needs its own copy)
    if atlas_img_path != 'none':
        atlas_img = nib.load(atlas_img_path)
        if max_voxels is not None:  # gather voxel within parcels without averaging
            time_series = gather_voxel_within_parcel(fmri_img, atlas_img, max_voxels)
        else:
            masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
            time_series = masker.fit_transform(fmri_img)
            time_series = time_series.T
    else:
        time_series = fmri_img.get_fdata()
    
    # time_series = time_series.T
    time_series = resample_fmri_linear(time_series, target_tr=2, source_tr=0.72)
    
    file_id = get_file_id(file_path, subject, session)
    
    return {
        'file_id': file_id,
        'subject': subject,
        'session': session,
        'file_path': file_path,
        'time_series': time_series,
        'success': True,
        'error': None
    }


def save_batch_to_hdf5(results, batch_idx):
    """Save a batch of results to HDF5 file."""
    with h5py.File(HDF5_FILE, 'a') as f:
        # Prepare batch metadata
        subjects = []
        sessions = []
        file_ids = []
        file_paths = []
        shapes = []
        dataset_names = []
        
        for i, result in enumerate(results):
            if result['success']:
                dataset_name = f"sample_{batch_idx * BATCH_SIZE + i:06d}"
                time_series = result['time_series']
                
                # Store time series data
                f.create_dataset(
                    f"time_series/{dataset_name}", 
                    data=time_series.astype(np.float32),
                    compression='gzip',
                )
                
                # Collect metadata for batch storage
                subjects.append(result['subject'])
                sessions.append(result['session'])
                file_ids.append(result['file_id'])
                file_paths.append(result['file_path'])
                shapes.append(time_series.shape)
                dataset_names.append(dataset_name)
        
        # Store metadata as datasets instead of attributes
        if dataset_names:  # Only if we have successful results
            meta_group = f.require_group("metadata")
            
            # Convert strings to bytes for HDF5 storage
            subjects_bytes = [s.encode('utf-8') for s in subjects]
            sessions_bytes = [s.encode('utf-8') for s in sessions]
            file_ids_bytes = [s.encode('utf-8') for s in file_ids]
            file_paths_bytes = [s.encode('utf-8') for s in file_paths]
            dataset_names_bytes = [s.encode('utf-8') for s in dataset_names]
            
            # Create or extend metadata datasets
            batch_start_idx = batch_idx * BATCH_SIZE
            for idx, (subj, sess, fid, fpath, shape, dname) in enumerate(zip(
                subjects_bytes, sessions_bytes, file_ids_bytes, file_paths_bytes, shapes, dataset_names_bytes)):
                
                sample_idx = batch_start_idx + idx
                
                # Store each metadata field in its own expandable dataset
                for field_name, data in [
                    ('subjects', subj),
                    ('sessions', sess), 
                    ('file_ids', fid),
                    ('file_paths', fpath),
                    ('dataset_names', dname)
                ]:
                    if field_name not in meta_group:
                        # Create expandable dataset
                        meta_group.create_dataset(field_name, (1,), maxshape=(None,), 
                                                dtype=h5py.string_dtype(encoding='utf-8'))
                        meta_group[field_name][0] = data
                    else:
                        # Extend existing dataset
                        meta_group[field_name].resize((sample_idx + 1,))
                        meta_group[field_name][sample_idx] = data
                
    return True

def update_metadata_csv(successful_results):
    """Update the metadata CSV file with new results."""
    try:
        new_metadata = []
        for result in successful_results:
            new_metadata.append({
                'file_id': result['file_id'],
                'subject': result['subject'],
                'session': result['session'],
                'file_path': result['file_path'],
            })
        
        new_df = pd.DataFrame(new_metadata)
        
        # Use append mode instead of reading entire file
        if METADATA_FILE.exists():
            # Append without header for existing file
            new_df.to_csv(METADATA_FILE, mode='a', header=False, index=False)
        else:
            # Create new file with header
            new_df.to_csv(METADATA_FILE, index=False)
            
        return True
        
    except Exception as e:
        logger.error(f"Error updating metadata CSV: {str(e)}")
        return False

# Collect all fMRI file paths and metadata
def run_preprocessing(args):
    """Main preprocessing function."""
    logger.info("Collecting fMRI file paths...")
    fMRI_file_paths = []
    metadata = []

    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        if os.path.isdir(subj_path):
            session = 'rfMRI_REST1_LR'
            sess_path = os.path.join(subj_path, session)
            func_path = os.path.join(sess_path, 'func')
            if os.path.isdir(func_path):
                for file in os.listdir(func_path):
                    if file.endswith('SmNprest_prep.nii.nii'):
                        file_path = os.path.join(func_path, file)
                        fMRI_file_paths.append(file_path)
                        metadata.append({'subject': subject, 'session': session})

    logger.info(f"Found {len(fMRI_file_paths)} fMRI files")

    # Load progress and filter out already processed files
    progress = load_progress()
    processed_file_ids = set(progress["processed_files"])

    # Filter out already processed files
    remaining_files = []
    remaining_metadata = []
    for file_path, meta in zip(fMRI_file_paths, metadata):
        file_id = get_file_id(file_path, meta['subject'], meta['session'])
        if file_id not in processed_file_ids:
            remaining_files.append(file_path)
            remaining_metadata.append(meta)

    logger.info(f"Already processed: {len(processed_file_ids)} files")
    logger.info(f"Remaining to process: {len(remaining_files)} files")

    if not remaining_files:
        logger.info("All files already processed!")
        return

    # Load and save the atlas to a temporary file for multiprocessing
    logger.info("Loading atlas...")
    if args.atlas_name == 'Schaefer':
        atlas_data = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
        atlas_img = nib.load(atlas_data['maps'])
    else:
        target_img = mean_img('/data/qneuromark/Data/HCP/Data_BIDS/Preprocess_Data/872158/rfMRI_REST1_LR/func/SmNprest_prep.nii.nii')
        if args.atlas_name == 'TianS3':
            atlas_img = nib.load('data/atlas/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_2mm.nii.gz')
            atlas_img = resample_to_img(atlas_img, target_img, interpolation='nearest')
            if args.gather_parcel_voxel: MAX_VOXELS = 259
        elif args.atlas_name == 'HCPex':
            atlas_img = nib.load('data/atlas/HCPex_2mm.nii')
            atlas_img = resample_to_img(atlas_img, target_img, interpolation='nearest')
        elif args.atlas_name == 'A424':
            atlas_img = nib.load('data/atlas/A424+2mm.nii.gz')
            atlas_img = resample_to_img(atlas_img, target_img, interpolation='nearest')
        elif args.atlas_name == 'none':
            atlas_img = None
        else:
            logger.error(f"Unsupported atlas name: {args.atlas_name}")
            return

    if atlas_img is not None:
        atlas_img_path = OUTPUT_DIR / "atlas_temp.nii.gz"
        nib.save(atlas_img, atlas_img_path)
    else:
        atlas_img_path = 'none'

    # Prepare arguments for multiprocessing
    process_args = [
        (file_path, meta['subject'], meta['session'], str(atlas_img_path), MAX_VOXELS)
        for file_path, meta in zip(remaining_files, remaining_metadata)
    ]

    # Process files in batches with multiprocessing
    logger.info(f"Starting processing with {N_PROCESSES} processes...")
    total_files = len(remaining_files)
    total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

    successful_count = 0
    failed_count = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_files)
        batch_args = process_args[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_args)} files)")
        
        # Process batch with multiprocessing
        with mp.Pool(processes=N_PROCESSES) as pool:
            batch_results = list(tqdm(
                pool.imap(process_single_file, batch_args),
                total=len(batch_args),
                desc=f"Batch {batch_idx + 1}"
            ))
        
        # Separate successful and failed results
        successful_results = [r for r in batch_results if r['success']]
        failed_results = [r for r in batch_results if not r['success']]
        
        successful_count += len(successful_results)
        failed_count += len(failed_results)
        
        # Save successful results to HDF5
        if successful_results:
            s = time.time()
            if save_batch_to_hdf5(successful_results, batch_idx):
                logger.info(f"Saved {len(successful_results)} files to HDF5 with {time.time() - s:.2f} seconds")
                
                # Update metadata CSV
                if update_metadata_csv(successful_results):
                    logger.info("Updated metadata CSV")
                
                # Update progress - only save count, not full list
                progress["total_processed"] += len(successful_results)
                # Only save file IDs for this batch to avoid memory bloat
                progress["last_batch_processed"] = [result['file_id'] for result in successful_results]
                progress["last_batch_idx"] = batch_idx
                save_progress(progress)
            else:
                logger.error(f"Failed to save batch {batch_idx}")
        
        # Log failed files
        for failed in failed_results:
            logger.error(f"Failed to process {failed['file_path']}: {failed['error']}")
        
        logger.info(f"Batch {batch_idx + 1} complete. Success: {len(successful_results)}, Failed: {len(failed_results)}")

    # Cleanup temporary atlas file
    if atlas_img_path.exists():
        atlas_img_path.unlink()

    # Final summary
    logger.info(f"Processing complete!")
    logger.info(f"Total successful: {successful_count}")
    logger.info(f"Total failed: {failed_count}")
    logger.info(f"Output saved to: {HDF5_FILE}")
    logger.info(f"Metadata saved to: {METADATA_FILE}")

# Create final consolidated numpy array if requested
def create_final_numpy_array(args):
    """Create final consolidated numpy array from HDF5 file."""
    logger.info("Creating final consolidated numpy array...")
    
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            time_series_group = f['time_series']
            dataset_names = sorted(time_series_group.keys())
            
            if not dataset_names:
                logger.warning("No data found in HDF5 file")
                return
            
            # Get dimensions from first dataset
            first_data = time_series_group[dataset_names[0]][:]
            num_rois = first_data.shape[1]
            
            # Find maximum timepoints
            max_timepoints = 0
            for name in dataset_names:
                data = time_series_group[name]
                max_timepoints = max(max_timepoints, data.shape[0])
            
            # Create consolidated array
            all_data = np.full((len(dataset_names), max_timepoints, num_rois), np.nan, dtype=np.float32)
            
            for idx, name in enumerate(tqdm(dataset_names, desc="Consolidating data")):
                data = time_series_group[name][:]
                all_data[idx, :data.shape[0], :] = data
            
            # Save consolidated array
            final_output = OUTPUT_DIR / f"HCP_fMRI_{args.atlas_name}_consolidated.npy"
            np.save(final_output, all_data)
            logger.info(f"Saved consolidated array to {final_output}")
            logger.info(f"Final shape: {all_data.shape}")
            
    except Exception as e:
        logger.error(f"Error creating consolidated array: {str(e)}")  


if __name__ == "__main__":
    main()