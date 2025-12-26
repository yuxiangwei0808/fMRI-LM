import argparse
import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import mean_img, resample_to_img
from nilearn.input_data import NiftiLabelsMasker
from tqdm import tqdm

from temporal_resample import resample_fmri_linear


@dataclass
class DatasetConfig:
    """Parameter bundle describing dataset-specific preprocessing quirks."""

    name: str
    default_root_dir: str
    default_output_dir: str
    default_hdf5_name: str
    metadata_filename: str
    default_atlas_name: str
    source_tr: float
    target_tr: float
    clip_length: Optional[int]
    standardize: bool
    gather_supported: bool
    default_max_voxels: Optional[int]
    file_finder: Callable[[str], Iterable[Tuple[str, str, str]]]


def _find_hcp_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    session = "rfMRI_REST1_LR"
    suffix = "SmNprest_prep.nii.nii"
    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        func_path = os.path.join(subj_path, session, "func")
        if not os.path.isdir(func_path):
            continue
        for file_name in sorted(os.listdir(func_path)):
            if file_name.endswith(suffix):
                yield os.path.join(func_path, file_name), subject, session


def _find_hcp_aging_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    suffix = "SmNSdmcf_prest.nii.nii"
    subdir = "func1_AP"
    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        for session in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, session)
            if not os.path.isdir(sess_path):
                continue
            func_path = os.path.join(sess_path, subdir)
            if not os.path.isdir(func_path):
                continue
            for file_name in sorted(os.listdir(func_path)):
                if file_name.endswith(suffix):
                    yield os.path.join(func_path, file_name), subject, session


def _find_ukb_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    suffix = "SmNprest_prep.nii.nii"
    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        for session in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, session)
            if not os.path.isdir(sess_path):
                continue
            func_path = os.path.join(sess_path, "func")
            if not os.path.isdir(func_path):
                continue
            for file_name in sorted(os.listdir(func_path)):
                if file_name.endswith(suffix):
                    yield os.path.join(func_path, file_name), subject, session

def _find_ehbs_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    suffix = "swarsfMRI.nii"
    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        func_path = os.path.join(subj_path, "rest")
        if os.path.isdir(func_path):
            for file_name in sorted(os.listdir(func_path)):
                if file_name.endswith(suffix):
                    yield os.path.join(func_path, file_name), subject, "ses01"

def _find_adni_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    metadata = pd.read_csv('data/ADNI/fmri/metadata.csv')

    for i, row in metadata.iterrows():
        yield row['fmri_path'], row['subject_id'], os.path.basename(os.path.dirname(row['session_path']))

def _find_abcd_files(root_dir: str):
    suffix = "SmNpdmcf_prest.nii"

    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        for session in sorted(os.listdir(subj_path)):
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

                    yield os.path.join(func_path, target_fmri_file[0]), subject, session
                    break  # Only take the first func_ folder

def _find_adhd_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    ref_df = pd.read_csv('data/ADHD200/fmri/metadata_with_text_medical.csv')
    ids = ref_df['subject_id'].tolist()
    ids = [str(i) for i in ids]

    for subject in sorted(os.listdir(root_dir)):
        iid = subject.split('_')[-2]

        if iid not in ids:
            continue
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue

        func_path = os.path.join(subj_path, "SM.nii")
        if not os.path.isfile(func_path):
            continue

        yield func_path, iid, "ses01"

def _find_abide2_files(root_dir: str) -> Iterable[Tuple[str, str, str]]:
    suffix = 'SM.nii'

    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue

        func_path = os.path.join(subj_path, suffix)
        if not os.path.isfile(func_path):
            continue

        yield func_path, subject, "ses01"

DATASETS: Dict[str, DatasetConfig] = {
    "hcp": DatasetConfig(
        name="hcp",
        default_root_dir="/data/qneuromark/Data/HCP/Data_BIDS/Preprocess_Data/",
        default_output_dir="./hcp_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="HCP_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=0.72,
        target_tr=2.0,
        clip_length=None,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_hcp_files,
    ),
    "hcp_aging": DatasetConfig(
        name="hcp_aging",
        default_root_dir="/data/neuromark2/Data/HCP_Aging/Data_BIDS/Raw_Data/",
        default_output_dir="./hcp_aging_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="HCP_Aging_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=0.8,
        target_tr=2.0,
        clip_length=None,
        standardize=True,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_hcp_aging_files,
    ),
    "ukb": DatasetConfig(
        name="ukb",
        default_root_dir="/data/qneuromark/Data/UKBiobank/Data_BIDS/Preprocess_Data/",
        default_output_dir="./ukb_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="UKBiobank_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=0.735,
        target_tr=2.0,
        clip_length=180,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_ukb_files,
    ),
    'ehbs': DatasetConfig(
        name='ehbs',
        default_root_dir="/data/qneuromark/Data/Emory_Healthy_Brain/rest/",
        default_output_dir="./ehbs_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="EHBS_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=0.8,
        target_tr=2.0,
        clip_length=200,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_ehbs_files,
    ),
    'adni': DatasetConfig(
        name='adni',
        default_root_dir="data/qneuromark/Data/ADNI/Updated/fMRI/ADNI/",
        default_output_dir="./adni_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="ADNI_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=2,
        target_tr=2.0,
        clip_length=None,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_adni_files,
    ),
    'abcd': DatasetConfig(
        name='abcd',
        default_root_dir='/data/neuromark2/Data/ABCD/Data_BIDS_5/Raw_Data/',
        default_output_dir="./abcd_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="ABCD_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=0.8,
        target_tr=2.0,
        clip_length=None,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_abcd_files,
    ),
    'adhd200': DatasetConfig(
        name='adhd200',
        default_root_dir='/data/qneuromark/Data/ADHD/ADHD200/ZN_Neuromark/ZN_Prep_fMRI/',
        default_output_dir="./adhd200_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="ADHD200_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=2.0,
        target_tr=2.0,
        clip_length=None,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_adhd_files,
    ),
    'abide2': DatasetConfig(
        name='abide2',
        default_root_dir='/data/qneuromark/Data/Autism/ABIDE2/ZN_Neuromark/ZN_Prep_fMRI/',
        default_output_dir="./abide2_preprocessing_output",
        default_hdf5_name="data_resampled.h5",
        metadata_filename="ABIDE2_fMRI_metadata.csv",
        default_atlas_name="TianS3",
        source_tr=2.0,
        target_tr=2.0,
        clip_length=None,
        standardize=False,
        gather_supported=False,
        default_max_voxels=None,
        file_finder=_find_abide2_files,
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified fMRI preprocessing for HCP, HCP Aging, and UKBiobank datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASETS.keys()),
        required=True,
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes to spawn (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=50,
        help="Number of samples saved per batch (default: 50).",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Create consolidated NumPy array from existing HDF5 output and exit.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Override dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--atlas-name",
        type=str,
        default='TianS3',
        help="Atlas name to use (default resolved per dataset).",
    )
    parser.add_argument(
        "--gather-parcel-voxel",
        action="store_true",
        default=False,
        help="For supported datasets, gather voxel-wise signals within each parcel instead of averaging.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="Maximum voxels kept per parcel when gather mode is enabled.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=None,
        help="Optional override for post-resampling temporal length (timepoints).",
    )
    parser.add_argument(
        "--hdf5-name",
        type=str,
        default=None,
        help="Optional override for HDF5 filename.",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default=None,
        help="Optional override for metadata CSV filename.",
    )
    parser.add_argument(
        "--standardize",
        type=str,
        choices=["auto", "true", "false"],
        default="false",
        help="Whether to z-score parcel signals (default: dataset-specific).",
    )
    return parser.parse_args()


def load_progress(progress_file: Path) -> Dict[str, object]:
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
        if "processed_files" not in progress:
            progress["processed_files"] = []
        return progress
    return {
        "processed_files": [],
        "total_processed": 0,
        "last_batch_processed": [],
        "last_batch_idx": -1,
    }


def save_progress(progress_file: Path, progress: Dict[str, object]) -> None:
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def append_string_dataset(group: h5py.Group, name: str, value: str) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    if name not in group:
        group.create_dataset(name, data=[value], maxshape=(None,), dtype=dtype)
    else:
        dataset = group[name]
        dataset.resize((dataset.shape[0] + 1,))
        dataset[-1] = value


def append_shape_dataset(group: h5py.Group, shape: Tuple[int, ...]) -> None:
    shape_arr = np.array(shape, dtype=np.int32)
    if "shapes" not in group:
        group.create_dataset("shapes", data=[shape_arr], maxshape=(None, shape_arr.size))
    else:
        dataset = group["shapes"]
        dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
        dataset[-1] = shape_arr


def gather_voxel_within_parcel(
    fmri_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    max_voxels: int,
) -> np.ndarray:
    fmri_data = fmri_img.get_fdata()
    atlas_data = atlas_img.get_fdata()

    labels = np.unique(atlas_data)
    labels = labels[labels != 0]

    rng = np.random.default_rng()
    parcel_series: List[np.ndarray] = []

    for label in labels:
        mask = atlas_data == label
        voxel_time_series = fmri_data[mask]  # (n_voxels, time)
        if voxel_time_series.size == 0:
            padding = np.full((max_voxels, fmri_data.shape[-1]), -1.0, dtype=np.float32)
            parcel_series.append(padding)
            continue
        if voxel_time_series.shape[0] > max_voxels:
            idx = rng.choice(voxel_time_series.shape[0], max_voxels, replace=False)
            voxel_time_series = voxel_time_series[idx]
        elif voxel_time_series.shape[0] < max_voxels:
            pad = np.full((max_voxels - voxel_time_series.shape[0], voxel_time_series.shape[1]), -1.0)
            voxel_time_series = np.vstack([voxel_time_series, pad])
        parcel_series.append(voxel_time_series.astype(np.float32))

    stacked = np.stack(parcel_series, axis=0)  # (n_parcels, max_voxels, time)
    flattened = stacked.reshape(stacked.shape[0] * stacked.shape[1], stacked.shape[2])
    return flattened.T  # (time, features)


def process_single_file(args: Tuple) -> Dict[str, object]:
    (
        file_path,
        subject,
        session,
        atlas_img_path,
        standardize,
        source_tr,
        target_tr,
        clip_length,
        gather_voxel,
        max_voxels,
        dataset_key,
    ) = args

    fmri_img = nib.load(file_path)

    if atlas_img_path == "none":
        data = fmri_img.get_fdata()
        time_series = data.reshape(-1, data.shape[-1]).T  # (time, voxels)
    else:
        atlas_img = nib.load(atlas_img_path)
        if gather_voxel:
            if max_voxels is None:
                raise ValueError("max_voxels must be set when gather_parcel_voxel is enabled")
            time_series = gather_voxel_within_parcel(fmri_img, atlas_img, max_voxels)
        else:
            masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=standardize)
            time_series = masker.fit_transform(fmri_img)  # (time, rois)

    # Resample expects time as last dimension
    if source_tr != target_tr:
        time_series = resample_fmri_linear(time_series.T, source_tr=source_tr, target_tr=target_tr).T

    assert time_series.shape[-1] == 450

    if clip_length is not None and time_series.shape[0] > clip_length:
        time_series = time_series[:clip_length]

    file_id = f"{subject}_{session}_{Path(file_path).name}"

    return {
        "file_id": file_id,
        "subject": subject,
        "session": session,
        "file_path": file_path,
        "dataset": dataset_key,
        "time_series": time_series.astype(np.float32).T,
        "success": True,
        "error": None,
    }


def save_batch_to_hdf5(
    hdf5_path: Path,
    batch_results: List[Dict[str, object]],
    batch_size: int,
    logger: logging.Logger,
) -> int:
    saved = 0
    with h5py.File(hdf5_path, "a") as f:
        time_group = f.require_group("time_series")
        meta_group = f.require_group("metadata")
        next_index = int(f.attrs.get("next_index", 0))

        for result in batch_results:
            dataset_name = f"sample_{next_index:06d}"
            data = result["time_series"]
            time_group.create_dataset(
                dataset_name,
                data=data,
                compression="lzf",
            )

            append_string_dataset(meta_group, "subjects", result["subject"])
            append_string_dataset(meta_group, "sessions", result["session"])
            append_string_dataset(meta_group, "file_ids", result["file_id"])
            append_string_dataset(meta_group, "file_paths", result["file_path"])
            append_string_dataset(meta_group, "dataset_names", dataset_name)
            append_shape_dataset(meta_group, data.shape)

            saved += 1
            next_index += 1

        f.attrs["next_index"] = next_index
    logger.info("Saved %d samples to %s", saved, hdf5_path)
    return saved


def update_metadata_csv(metadata_path: Path, successful_results: List[Dict[str, object]], logger: logging.Logger) -> None:
    if not successful_results:
        return
    records = []
    for result in successful_results:
        data = result["time_series"]
        records.append(
            {
                "file_id": result["file_id"],
                "subject": result["subject"],
                "session": result["session"],
                "file_path": result["file_path"],
                "shape_time": data.shape[1],
                "shape_features": data.shape[0] if data.ndim > 1 else 1,
            }
        )
    df = pd.DataFrame.from_records(records)
    if metadata_path.exists():
        df.to_csv(metadata_path, mode="a", header=False, index=False)
    else:
        df.to_csv(metadata_path, index=False)
    logger.info("Metadata CSV updated at %s", metadata_path)


def prepare_atlas(
    atlas_name: str,
    dataset_key: str,
    output_dir: Path,
    reference_img: Optional[nib.Nifti1Image],
    logger: logging.Logger,
) -> str:
    if atlas_name == "none":
        return "none"

    if atlas_name == "Schaefer":
        atlas_data = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
        atlas_img = nib.load(atlas_data["maps"])
    elif atlas_name == "TianS3":
        atlas_img = nib.load(
            "data/atlas/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_2mm.nii.gz"
        )
    elif atlas_name == "HCPex":
        atlas_img = nib.load("data/atlas/HCPex_2mm.nii")
    elif atlas_name == "A424":
        atlas_img = nib.load("data/atlas/A424+2mm.nii.gz")
    else:
        raise ValueError(f"Unsupported atlas name: {atlas_name}")

    if reference_img is not None:
        atlas_img = resample_to_img(atlas_img, reference_img, interpolation="nearest")

    atlas_path = output_dir / f"atlas_{dataset_key}.nii.gz"
    nib.save(atlas_img, atlas_path)
    logger.info("Atlas saved to %s", atlas_path)
    return str(atlas_path)


def create_final_numpy_array(
    hdf5_path: Path,
    output_dir: Path,
    atlas_name: str,
    logger: logging.Logger,
) -> None:
    logger.info("Creating consolidated NumPy array from %s", hdf5_path)
    if not hdf5_path.exists():
        logger.error("HDF5 file %s does not exist", hdf5_path)
        return

    with h5py.File(hdf5_path, "r") as f:
        if "time_series" not in f:
            logger.error("No time_series group in %s", hdf5_path)
            return
        time_group = f["time_series"]
        dataset_names = sorted(time_group.keys())
        if not dataset_names:
            logger.warning("No datasets found inside time_series group")
            return
        first = time_group[dataset_names[0]][:]
        feature_dim = first.shape[1]
        max_time = max(time_group[name].shape[0] for name in dataset_names)

        stacked = np.full((len(dataset_names), max_time, feature_dim), np.nan, dtype=np.float32)
        for idx, name in enumerate(tqdm(dataset_names, desc="Consolidating")):
            data = time_group[name][:]
            stacked[idx, : data.shape[0], : data.shape[1]] = data

    output_path = output_dir / f"fMRI_{atlas_name}_consolidated.npy"
    np.save(output_path, stacked)
    logger.info("Consolidated array saved to %s with shape %s", output_path, stacked.shape)


def configure_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"preprocessing_{log_file.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    args = parse_arguments()
    config = DATASETS[args.dataset]

    root_dir = args.root_dir or config.default_root_dir
    output_dir = Path(args.output_dir or config.default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_file = output_dir / (args.hdf5_name or config.default_hdf5_name)
    metadata_file = output_dir / (args.metadata_name or config.metadata_filename)
    progress_file = output_dir / f"progress_{config.name}.json"
    error_log_file = output_dir / f"errors_{config.name}.log"

    logger = configure_logger(error_log_file)

    n_processes = args.processes if args.processes else min(8, mp.cpu_count())
    atlas_name = args.atlas_name or config.default_atlas_name
    batch_size = args.batch_size

    if args.standardize == "auto":
        standardize = config.standardize
    else:
        standardize = args.standardize == "true"

    clip_length = args.clip_length if args.clip_length is not None else config.clip_length

    if args.gather_parcel_voxel and not config.gather_supported:
        logger.warning("Dataset %s does not support gather_parcel_voxel; ignoring flag.", config.name)
        gather_parcel_voxel = False
    else:
        gather_parcel_voxel = args.gather_parcel_voxel

    max_voxels = args.max_voxels if args.max_voxels is not None else config.default_max_voxels
    if gather_parcel_voxel and max_voxels is None:
        raise ValueError("max_voxels must be provided for gather_parcel_voxel mode")

    logger.info("Dataset: %s", config.name)
    logger.info("Root dir: %s", root_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("HDF5 file: %s", hdf5_file)
    logger.info("Metadata file: %s", metadata_file)
    logger.info("Processes: %d", n_processes)

    if args.consolidate:
        create_final_numpy_array(hdf5_file, output_dir, atlas_name, logger)
        return

    logger.info("Collecting fMRI files...")
    files = list(config.file_finder(root_dir))
    logger.info("Discovered %d candidate files", len(files))
    if not files:
        logger.error("No matching fMRI files discovered under %s", root_dir)
        return

    progress = load_progress(progress_file)
    processed_file_ids = set(progress.get("processed_files", []))

    remaining_files: List[Tuple[str, str, str]] = []
    for file_path, subject, session in files:
        file_id = f"{subject}_{session}_{Path(file_path).name}"
        if file_id not in processed_file_ids:
            remaining_files.append((file_path, subject, session))

    logger.info("Already processed: %d", len(files) - len(remaining_files))
    logger.info("Remaining: %d", len(remaining_files))

    if not remaining_files:
        logger.info("All files already processed. Exiting.")
        return

    reference_img = None
    reference_img = mean_img(remaining_files[0][0])

    atlas_img_path = prepare_atlas(atlas_name, config.name, output_dir, reference_img, logger)

    params = (
        standardize,
        config.source_tr,
        config.target_tr,
        clip_length,
        gather_parcel_voxel,
        max_voxels,
        config.name,
    )

    total_files = len(remaining_files)
    total_batches = (total_files + batch_size - 1) // batch_size
    total_success = 0
    total_failed = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_files)
        batch = remaining_files[start:end]
        logger.info("Processing batch %d/%d with %d files", batch_idx + 1, total_batches, len(batch))

        process_args = [
            (
                file_path,
                subject,
                session,
                atlas_img_path,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6],
            )
            for file_path, subject, session in batch
        ]

        with mp.Pool(processes=n_processes) as pool:
            batch_results = list(
                tqdm(pool.imap(process_single_file, process_args), total=len(process_args), desc=f"Batch {batch_idx + 1}")
            )

        successful = [res for res in batch_results if res["success"]]
        failed = [res for res in batch_results if not res["success"]]

        if successful:
            saved = save_batch_to_hdf5(hdf5_file, successful, batch_size, logger)
            update_metadata_csv(metadata_file, successful, logger)
            processed_file_ids.update(res["file_id"] for res in successful)
            progress["processed_files"] = list(processed_file_ids)
            progress["total_processed"] = progress.get("total_processed", 0) + saved
            progress["last_batch_processed"] = [res["file_id"] for res in successful]
            progress["last_batch_idx"] = batch_idx
            save_progress(progress_file, progress)

        for res in failed:
            logger.error("Failed to process %s: %s", res["file_path"], res["error"])

        total_success += len(successful)
        total_failed += len(failed)
        logger.info("Batch %d complete. Success: %d, Failed: %d", batch_idx + 1, len(successful), len(failed))

    if atlas_img_path != "none":
        try:
            Path(atlas_img_path).unlink()
            logger.info("Removed temporary atlas at %s", atlas_img_path)
        except OSError:
            logger.debug("Could not remove atlas file %s", atlas_img_path)

    logger.info("Processing finished. Total success: %d, total failed: %d", total_success, total_failed)
    logger.info("Outputs -> HDF5: %s, Metadata: %s", hdf5_file, metadata_file)


if __name__ == "__main__":
    main()
