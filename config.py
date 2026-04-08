"""
Configuration for the RL-based user scheduling and beamforming system.
"""
import numpy as np
import os

# System parameters
num_users = 50
num_total_samples = 70000
num_training_samples = 60000
num_val_samples = 1000
num_test_samples = 2000

K = 20   # Maximum number of users supported by BS
M = 30   # Maximum uplink feedback capacity (nomination budget)
Nt = 32  # Number of transmit antennas
Nr = 1
Nw = Nt  # Number of DFT beams

np.random.seed(2024)

# SNR and noise
SNR_dB = 15
noise_pwr = 1 / (10 ** (0.1 * SNR_dB))

# Learning parameters
num_epochs = 150
batch_size = 192
my_learning_rate = 0.001
Num_save_m_file = num_epochs
save_epoch_start = num_epochs // 2

num_experiments = num_test_samples

# Angle grid and DFT matrix
anglegrid_prob = np.zeros(Nw)
for a in range(Nw):
    anglegrid_prob[a] = (2 / Nw) * a - 1

Aw = np.zeros((Nt, Nt), dtype=complex)
for idx in range(Nt):
    Aw[:, idx] = 1 / np.sqrt(Nt) * np.exp(1j * np.pi * np.arange(Nt) * anglegrid_prob[idx])

# Paths (relative to project root)
save_path = "./result"
save_parameter_path = save_path + "/save_parameter"
save_parameter_file_name = save_parameter_path + "/Ne{}".format(num_epochs)
save_model_path = save_path + "/save_model"
save_model_file_name = save_model_path + "/38901_UPA/Ne{}".format(num_epochs)
save_fig_path = save_path + "/save_fig"
save_fig_folder_path = save_fig_path + "/Ne{}".format(num_epochs)
save_testresult_path = save_path + "/save_testresult"
save_testresult_file_name = save_testresult_path + "/Exp{}".format(num_experiments)


def ensure_directory_exists(directory_path, verbose=False):
    """Create directory if it doesn't exist."""
    if not os.path.isdir(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
            if verbose:
                print(f"Created directory: {directory_path}")
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")
            raise e


def ensure_file_directory_exists(file_path, verbose=True):
    """Create the directory for a file path if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory:
        ensure_directory_exists(directory, verbose=verbose)
