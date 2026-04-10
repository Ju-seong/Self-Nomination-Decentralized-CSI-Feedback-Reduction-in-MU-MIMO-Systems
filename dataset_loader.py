"""
Data loaders for channel datasets (RF and 3GPP-style channel models).
Set DATASET_ROOT to the directory that contains the dataset files when using
UMi_UPA, UMi_ULA, Berlin_UPA, or RMa_UPA.
"""
from pathlib import Path
import os

import h5py
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import DataLoader, Dataset

from config import *
import config


def _resolve_dataset_root():
    """Return a dataset directory if one can be inferred, else None."""
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        return Path(env_root).expanduser()

    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir / "datasets",
        this_dir.parent / "datasets",
        Path.cwd() / "datasets",
        Path.cwd().parent / "datasets",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


DATASET_ROOT = _resolve_dataset_root()
DATASET_SUBDIR = Path("Data_Narrowband(CH+UEposition)_Nt32_Nr1")
DATASET_MODE_TOKENS = {
    "UMi_UPA": ("umi", "upa"),
    "UMi_ULA": ("umi", "ula"),
    "Berlin_UPA": ("berlin", "upa"),
    "RMa_UPA": ("rma",),
}


def _dataset_path(relative_path):
    if DATASET_ROOT is None:
        raise FileNotFoundError(
            "Dataset root not found. Set DATASET_ROOT to the dataset directory "
            "before running a non-RF experiment."
        )
    return DATASET_ROOT / relative_path


def _find_dataset_file(channel_mode):
    """Locate the dataset file for the requested channel mode without hardcoding filenames."""
    dataset_dir = _dataset_path(DATASET_SUBDIR)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    tokens = DATASET_MODE_TOKENS[channel_mode]
    candidates = sorted(path for path in dataset_dir.glob("*.mat") if path.is_file())
    matches = []
    for path in candidates:
        path_name = path.name.lower()
        if all(token in path_name for token in tokens):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"No dataset file found for channel_mode={channel_mode} under {dataset_dir}. "
            "Set DATASET_ROOT to the correct dataset location if needed."
        )

    if len(matches) > 1:
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    return matches[0]


class CommunicationDataset(Dataset):
    """
    Channel dataset for different channel modes.
    channel_mode: "RF", "UMi_UPA", "UMi_ULA", "Berlin_UPA", or "RMa_UPA"
    """

    def __init__(self, channel_mode):
        if channel_mode == "RF":
            current_num_users = config.num_users
            self.H = np.sqrt(1 / 2) * (
                np.random.randn(num_total_samples, current_num_users, 1, Nt)
                + 1j * np.random.randn(num_total_samples, current_num_users, 1, Nt)
            )
        elif channel_mode == "UMi_UPA":
            dataset_path = _find_dataset_file(channel_mode)
            dataset = spio.loadmat(dataset_path)
            H_tmp = dataset["H_set"]
            self._load_and_normalize(H_tmp)
        elif channel_mode == "UMi_ULA":
            dataset_path = _find_dataset_file(channel_mode)
            dataset = spio.loadmat(dataset_path)
            H_tmp = dataset["H_set"]
            self._load_and_normalize(H_tmp)
        elif channel_mode == "Berlin_UPA":
            dataset_path = _find_dataset_file(channel_mode)
            dataset = spio.loadmat(dataset_path)
            H_tmp = dataset["H_set"]
            self._load_and_normalize(H_tmp)
        elif channel_mode == "RMa_UPA":
            dataset_path = _find_dataset_file(channel_mode)
            with h5py.File(dataset_path, "r") as f:
                H_set_data = np.array(f["H_set"]).T
                H_tmp = H_set_data["real"] + 1j * H_set_data["imag"]
            current_num_users = config.num_users
            all_indices = np.arange(H_tmp.shape[0])
            selected_indices = []
            num_repeats = (num_total_samples * current_num_users // H_tmp.shape[0]) + 1
            for _ in range(num_repeats):
                np.random.shuffle(all_indices)
                selected_indices.extend(all_indices)
            selected_indices = np.array(selected_indices[: num_total_samples * current_num_users])
            selected_indices = selected_indices.reshape(num_total_samples, current_num_users)
            self.H = H_tmp[selected_indices][:, :, :Nt].reshape(
                num_total_samples, current_num_users, 1, Nt
            )
            H_reshaped = self.H.reshape(-1, 1, Nt)
            fro_norms = np.linalg.norm(H_reshaped, axis=(1, 2))
            fro_norm_average = np.mean(fro_norms)
            self.H *= np.sqrt(Nt) / fro_norm_average
        else:
            raise ValueError(
                f"Unsupported channel_mode: {channel_mode}. "
                "Expected 'RF', 'UMi_UPA', 'UMi_ULA', 'Berlin_UPA', or 'RMa_UPA'."
            )

        self.num_samples = self.H.shape[0]
        self.num_users = self.H.shape[1]

    def _load_and_normalize(self, H_tmp):
        all_indices = np.arange(H_tmp.shape[0])
        selected_indices = []
        num_repeats = (num_total_samples * num_users // H_tmp.shape[0]) + 1
        for _ in range(num_repeats):
            np.random.shuffle(all_indices)
            selected_indices.extend(all_indices)
        selected_indices = np.array(selected_indices[: num_total_samples * num_users])
        selected_indices = selected_indices.reshape(num_total_samples, num_users)
        self.H = H_tmp[selected_indices][:, :, :Nt].reshape(
            num_total_samples, num_users, 1, Nt
        )
        H_reshaped = self.H.reshape(-1, 1, Nt)
        fro_norms = np.linalg.norm(H_reshaped, axis=(1, 2))
        fro_norm_average = np.mean(fro_norms)
        self.H *= np.sqrt(Nt) / fro_norm_average

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.H[idx]


def create_dataloader(channel_mode, batch_size=64, shuffle=True, num_workers=4):
    dataset = CommunicationDataset(channel_mode)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
