"""
Training script for RL-based user scheduling and beamforming.
"""
import os
import argparse
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from loaders import CommunicationDataset
from config import *
from learning_modules import get_module_class


def get_model_class(method, input_type):
    """Return the model class for the given method and input type."""
    return get_module_class(method, input_type)


def get_model_save_path(method, input_type, scheduling, beamforming, channel_mode):
    """Generate model save path from run configuration."""
    channel_dir = channel_mode if channel_mode in ("RF", "UMi_UPA", "UMi_ULA", "Berlin_UPA", "RMa_UPA") else "UMi_UPA"
    base = f"{save_model_path}/{channel_dir}/Ne{num_epochs}"
    method_suffix = "REINFORCE" if method == "reinforce" else "DirectGrad"
    input_suffix = "inH" if input_type == "full" else "inChg"
    sched_suffix = "Random" if scheduling == "random" else "Greedy"
    bf_suffix = "ZF" if beamforming == "zf" else "RZF"
    return f"{base}_Nt{Nt}_UE{num_users}_M{M}_K{K}_{method_suffix}_{input_suffix}_{sched_suffix}_{bf_suffix}_{channel_dir}_best.pth"


def get_parameter_filename(method, input_type, scheduling, beamforming, channel_mode):
    """Generate parameter .mat filename from run configuration."""
    channel_dir = channel_mode if channel_mode in ("RF", "UMi_UPA", "UMi_ULA", "Berlin_UPA", "RMa_UPA") else "UMi_UPA"
    base = f"{save_parameter_path}/{channel_dir}/Ne{num_epochs}"
    method_suffix = "REINFORCE" if method == "reinforce" else "DirectGrad"
    input_suffix = "inH" if input_type == "full" else "inChg"
    sched_suffix = "Random" if scheduling == "random" else "Greedy"
    bf_suffix = "ZF" if beamforming == "zf" else "RZF"
    return f"{base}_UE{num_users}_M{M}_K{K}_{method_suffix}_{input_suffix}_{sched_suffix}_{bf_suffix}_{channel_dir}_best.mat"


def main():
    parser = argparse.ArgumentParser(description="RL-based user scheduling and beamforming training")
    parser.add_argument("--method", type=str, choices=["reinforce", "directgrad"], default="reinforce")
    parser.add_argument("--input_type", type=str, choices=["full", "chg_input"], default="full")
    parser.add_argument("--scheduling", type=str, choices=["random", "greedy"], default="greedy")
    parser.add_argument("--beamforming", type=str, choices=["zf", "rzf"], default="rzf")
    parser.add_argument(
        "--channel_mode",
        type=str,
        choices=["RF", "UMi_UPA", "UMi_ULA", "Berlin_UPA", "RMa_UPA"],
        default="RMa_UPA",
    )
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ModelClass = get_model_class(args.method, args.input_type)
    model_save_path = get_model_save_path(
        args.method, args.input_type, args.scheduling, args.beamforming, args.channel_mode
    )
    parameter_filename = get_parameter_filename(
        args.method, args.input_type, args.scheduling, args.beamforming, args.channel_mode
    )

    print(f"Training: method={args.method}, input_type={args.input_type}, scheduling={args.scheduling}, beamforming={args.beamforming}, channel={args.channel_mode}")
    print(f"Model path: {model_save_path}")

    dataset = CommunicationDataset(args.channel_mode)
    train_dataset = Subset(dataset, range(num_training_samples))
    val_dataset = Subset(dataset, range(num_training_samples, num_training_samples + num_val_samples))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ModelClass(scheduling_method=args.scheduling, beamforming_method=args.beamforming).to(device)

    loss_arr = np.zeros((num_epochs, 1))
    sum_rate_arr = np.zeros((num_epochs, 1))
    avg_num_ones_arr = np.zeros((num_epochs, 1))
    val_sum_rate_arr = np.zeros((num_epochs, 1))
    val_avg_num_ones_arr = np.zeros((num_epochs, 1))
    best_sum_rate = -float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_sum_rate = total_avg_num_ones = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for H_batch in pbar:
            H_batch = H_batch.to(device)
            loss_tmp, sum_rate_tmp, avg_ones_tmp = model.train_module(H_batch, training=True)
            total_loss += loss_tmp
            total_sum_rate += sum_rate_tmp
            total_avg_num_ones += avg_ones_tmp
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        if hasattr(model, "dual_var"):
            print(f"  Dual var: {model.dual_var.item():.2f}, Sum rate: {sum_rate_tmp:.2f}, Nominated: {avg_ones_tmp:.2f}")
        else:
            print(f"  Sum rate: {sum_rate_tmp:.2f}, Nominated: {avg_ones_tmp:.2f}")

        model.eval()
        val_sum_rate = val_avg_ones = 0.0
        with torch.no_grad():
            for H_batch in val_loader:
                H_batch = H_batch.to(device)
                _, sr, no = model.train_module(H_batch, training=False)
                val_sum_rate += sr
                val_avg_ones += no
        val_avg_sum_rate = val_sum_rate / len(val_loader)
        val_avg_ones_final = val_avg_ones / len(val_loader)
        print(f"  Val Sum Rate: {val_avg_sum_rate:.4f}, Val Nominated: {val_avg_ones_final:.4f}")

        if epoch > save_epoch_start and val_avg_sum_rate > best_sum_rate and avg_ones_tmp <= M:
            best_sum_rate = val_avg_sum_rate
            ensure_file_directory_exists(model_save_path, verbose=False)
            torch.save(model.state_dict(), model_save_path)
            print(f"  Best model saved at epoch {epoch+1} (sum rate {best_sum_rate:.4f})")

        n_batches = num_training_samples // batch_size
        loss_arr[epoch] = total_loss / n_batches
        sum_rate_arr[epoch] = total_sum_rate / n_batches
        avg_num_ones_arr[epoch] = total_avg_num_ones / n_batches
        val_sum_rate_arr[epoch] = val_avg_sum_rate
        val_avg_num_ones_arr[epoch] = val_avg_ones_final

    ensure_file_directory_exists(parameter_filename, verbose=False)
    spio.savemat(
        parameter_filename,
        {
            "Out_epochs": num_epochs,
            "sum_rate": sum_rate_arr,
            "loss": loss_arr,
            "avg_num_ones": np.abs(avg_num_ones_arr),
            "val_sum_rate": val_sum_rate_arr,
            "val_avg_num_ones": np.abs(val_avg_num_ones_arr),
            "learning_rate": my_learning_rate,
            "batch_size": batch_size,
            "method": args.method,
            "input_type": args.input_type,
            "scheduling": args.scheduling,
            "beamforming": args.beamforming,
        },
    )
    print(f"Training done. Model: {model_save_path}")


if __name__ == "__main__":
    main()
