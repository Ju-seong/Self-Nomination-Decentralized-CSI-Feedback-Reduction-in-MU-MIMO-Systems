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


def _dataset_path(relative_path):
    if DATASET_ROOT is None:
        raise FileNotFoundError(
            "Dataset root not found. Set DATASET_ROOT to the dataset directory "
            "before running a non-RF experiment."
        )
    return DATASET_ROOT / relative_path


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
            dataset_path = _dataset_path(
                "Data_Narrowband(CH+UEposition)_Nt32_Nr1/"
                "AUTO_DISCOVERED_UMI_UPA_DATASET.mat"
            )
            dataset = spio.loadmat(dataset_path)
            H_tmp = dataset["H_set"]
            self._load_and_normalize(H_tmp)
        elif channel_mode == "UMi_ULA":
            dataset_path = _dataset_path(
                "Data_Narrowband(CH+UEposition)_Nt32_Nr1/"
                "AUTO_DISCOVERED_UMI_ULA_DATASET.mat"
            )
            dataset = spio.loadmat(dataset_path)
            H_tmp = dataset["H_set"]
            self._load_and_normalize(H_tmp)
        elif channel_mode == "Berlin_UPA":
            dataset_path = _dataset_path(
                "Data_Narrowband(CH+UEposition)_Nt32_Nr1/"
                "AUTO_DISCOVERED_BERLIN_UPA_DATASET.mat"
            )
            dataset = spio.loadmat(dataset_path)
            H_tmp = dataset["H_set"]
            self._load_and_normalize(H_tmp)
        elif channel_mode == "RMa_UPA":
            dataset_path = _dataset_path(
                "Data_Narrowband(CH+UEposition)_Nt32_Nr1/"
                "AUTO_DISCOVERED_RMA_UPA_DATASET.mat"
            )
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
