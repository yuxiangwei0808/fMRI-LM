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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UKBiobank fMRI preprocessing with multiprocessing")
    parser.add_argument("--processes", type=int, default=1,
                       help="Number of processes to use (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for saving data (default: 50)")
    parser.add_argument("--consolidate", action="store_true",
                       help="Only create consolidated numpy array from existing HDF5 data")
    parser.add_argument('--dataset_name', type=str, default='HCP', choices=['UKB', 'HCP', 'ABCD'],
                       help='Dataset name to process (default: HCP)')
    parser.add_argument("--output-dir", type=str, default="./preprocessing_output",
                       help="Output directory for processed data")
    parser.add_argument('--atlas_name', type=str, default='Schaefer_400_7',
                       help='Atlas name to use for preprocessing (default: Schaefer)')

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
logger = None

SOURCE_TR = None
VALID_LEN_TIMEPOINTS = None  # minmum number of timepoints after resampling to 2s TR

def main():
    args = parse_arguments()

    if args.dataset_name == 'UKB':
        args.root_dir = "/data/qneuromark/Data/UKBiobank/Data_BIDS/Preprocess_Data/"
    elif args.dataset_name == 'HCP':
        args.root_dir = "/data/qneuromark/Data/HCP/Data_BIDS/Preprocess_Data/"
    elif args.dataset_name == 'ABCD':
        args.root_dir = "/data/neuromark2/Data/ABCD/Data_BIDS_5/Raw_Data/"
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
    
    # Update global configuration based on arguments
    global OUTPUT_DIR, BATCH_SIZE, N_PROCESSES, root_dir
    global PROGRESS_FILE, ERROR_LOG_FILE, HDF5_FILE, METADATA_FILE, logger, SOURCE_TR, VALID_LEN_TIMEPOINTS

    if args.dataset_name == 'UKB':
        SOURCE_TR = 0.735  # UKB TR
        VALID_LEN_TIMEPOINTS = 180
    elif args.dataset_name == 'HCP':
        SOURCE_TR = 0.72  # HCP TR
        VALID_LEN_TIMEPOINTS = 320
    elif args.dataset_name == 'ABCD':
        SOURCE_TR = 0.8  # ABCD TR
        VALID_LEN_TIMEPOINTS = 140
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
    
    OUTPUT_DIR = Path(args.output_dir)
    BATCH_SIZE = args.batch_size
    N_PROCESSES = args.processes if args.processes else min(8, mp.cpu_count())
    root_dir = args.root_dir
    
    # Update file paths
    PROGRESS_FILE = OUTPUT_DIR / "progress.json"
    ERROR_LOG_FILE = OUTPUT_DIR / "errors.log"
    HDF5_FILE = OUTPUT_DIR / f"data_fc.h5"
    METADATA_FILE = OUTPUT_DIR / "fMRI_metadata.csv"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
                time_series = result['fc']
                
                # Store fc data
                f.create_dataset(
                    f"fc/{dataset_name}", 
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


def process_single_file(args):
    """Process a single fMRI file and return time series data."""
    file_path, subject, session, atlas_img_path, net2idx, networks = args
    
    atlas_img = nib.load(atlas_img_path)
    fmri_img = nib.load(file_path)

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True, detrend=True, t_r=SOURCE_TR)
    ts = masker.fit_transform(fmri_img)

    corr_matrix = np.corrcoef(ts.T)  # shape (n_parcels, n_parcels)
    np.fill_diagonal(corr_matrix, 1)

    file_id = get_file_id(file_path, subject, session)

    return {
        'file_id': file_id,
        'subject': subject,
        'session': session,
        'file_path': file_path,
        'fc': corr_matrix,
        'success': True,
        'error': None
    }


def find_fmri_ukb(root_dir):
    """Find all fMRI files in UKB dataset."""
    fMRI_file_paths = []
    metadata = []

    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        if os.path.isdir(subj_path):
            for session in os.listdir(subj_path):
                sess_path = os.path.join(subj_path, session)
                func_path = os.path.join(sess_path, 'func')
                if os.path.isdir(func_path):
                    for file in os.listdir(func_path):
                        if file.endswith('SmNprest_prep.nii.nii'):
                            file_path = os.path.join(func_path, file)
                            fMRI_file_paths.append(file_path)
                            metadata.append({'subject': subject, 'session': session})

    return fMRI_file_paths, metadata

def find_fmri_hcp(root_dir):
    """Find all fMRI files in UKB dataset."""
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
    return fMRI_file_paths, metadata

def find_fmri_abcd(root_dir):
    """Find all fMRI files in ABCD dataset."""
    suffix = "SmNpdmcf_prest.nii"
    fMRI_file_paths = []
    metadata = []

    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        
        for session in os.listdir(subj_path):
            sess_path = os.path.join(subj_path, session)
            if not os.path.isdir(sess_path):
                continue

            for func in reversed(sorted(os.listdir(sess_path))):
                if func.startswith('func_'):
                    func_path = os.path.join(sess_path, func)
                    if not os.path.isdir(func_path):
                        continue
                    
                    target_fmri_file = [f for f in os.listdir(func_path) if f.endswith(suffix)]
                    
                    assert len(target_fmri_file) <= 1
                    if not target_fmri_file:
                        continue

                    file_path = os.path.join(func_path, target_fmri_file[0])
                    fMRI_file_paths.append(file_path)
                    metadata.append({'subject': subject, 'session': session})
                    break  # only process the latest func_*
                    
    return fMRI_file_paths, metadata


# Collect all fMRI file paths and metadata
def run_preprocessing(args):
    """Main preprocessing function."""
    logger.info("Collecting fMRI file paths...")
    fMRI_file_paths = []
    metadata = []

    if args.dataset_name == 'UKB':
        fMRI_file_paths, metadata = find_fmri_ukb(root_dir)
    elif args.dataset_name == 'HCP':
        fMRI_file_paths, metadata = find_fmri_hcp(root_dir)
    elif args.dataset_name == 'ABCD':
        fMRI_file_paths, metadata = find_fmri_abcd(root_dir)
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

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
    
    if args.dataset_name == 'UKB':
        target_img_path = '/data/qneuromark/Data/UKBiobank/Data_BIDS/Preprocess_Data/1001995/ses_01/func/SmNprest_prep.nii.nii'
    elif args.dataset_name == 'HCP':
        target_img_path = '/data/qneuromark/Data/HCP/Data_BIDS/Preprocess_Data/872158/rfMRI_REST1_LR/func/SmNprest_prep.nii.nii'
    elif args.dataset_name == 'ABCD':
        target_img_path = '/data/neuromark2/Data/ABCD/Data_BIDS_5/Raw_Data/NDARINVBGYRJERP/Baseline/func_20170128115213/SmNpdmcf_prest.nii'
    target_img = nib.load(target_img_path)

    # Load and save the atlas to a temporary file for multiprocessing
    logger.info("Loading atlas...")

    if args.atlas_name == 'Schaefer_400_7':
        atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, yeo_networks=7)
        atlas_img = nib.load(atlas['maps'])
        atlas_labels = atlas['labels']
        atlas_img = resample_to_img(atlas_img, target_img, interpolation='nearest')
    else:
        logger.error(f"Unsupported atlas name: {args.atlas_name}")
        return

    network_names = [lbl.decode('utf-8').split('_')[2] for lbl in atlas_labels]
    networks = sorted(set(network_names))
    
    net2idx = {net: [i for i, netname in enumerate(network_names) if netname == net]
            for net in networks}

    atlas_img_path = OUTPUT_DIR / "atlas_temp.nii.gz"
    nib.save(atlas_img, atlas_img_path)

    # Prepare arguments for multiprocessing
    process_args = [
        (file_path, meta['subject'], meta['session'], str(atlas_img_path), net2idx, networks)
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
        assert len(failed_results) == 0, "Some files failed to process. Check the error log."
        
        # Log failed results
        for failed_result in failed_results:
            logger.error(f"Failed to process {failed_result['file_id']}: {failed_result['error']}")
        
        successful_count += len(successful_results)
        failed_count += len(failed_results)
        
        # Save successful results to CSV
        if successful_results:
            save_batch_to_hdf5(successful_results, batch_idx)
        
        # Update progress
        processed_file_ids = [r['file_id'] for r in successful_results]
        progress["processed_files"].extend(processed_file_ids)
        progress["total_processed"] += len(successful_results)
        progress["last_batch_processed"] = processed_file_ids
        progress["last_batch_idx"] = batch_idx
        save_progress(progress)
        
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


if __name__ == "__main__":
    main()