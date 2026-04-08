import torch
import torch.nn as nn
import itertools
from config import *
from typing import Tuple
import numpy as np


def exhaustive_scheduling_batched(
    H_batch: torch.Tensor,
    K: int,
    beamforming_method: str = 'zf',
    noise_pwr: float = 1e-3,
    chunk_size: int = 1024,   # tune based on your GPU memory
):
    """
    Exhaustively search user subsets (sizes 1..K) for each sample in a batch,
    evaluating (B x combos) in parallel on GPU via flattening and chunking.

    Args
    ----
    H_batch : torch.Tensor
        [B, U, Nt] or [B, U, 1, Nt] (the singleton dim will be squeezed).
    K : int
        Max number of users to schedule.
    beamforming_method : str
        'zf' or 'rzf'.
    noise_pwr : float
        Noise power.
    chunk_size : int
        Number of subset-combinations to process per chunk (per m). Increase for more parallelism,
        decrease if you hit OOM.

    Returns
    -------
    best_masks : torch.Tensor
        [B, U] one-hot mask of selected users per batch sample.
    best_sum_rates : torch.Tensor
        [B] best sum-rate per sample.
    """
    from baseline_methods.sched_bf_modules import zf_beamforming, rzf_beamforming, sum_rate_calculation

    # Normalize shape
    assert H_batch.dim() in (3, 4), "H_batch must be [B,U,Nt] or [B,U,1,Nt]"
    if H_batch.dim() == 4:
        H_batch = H_batch.squeeze(2)  # [B, U, Nt]
    B, U, Nt = H_batch.shape
    device = H_batch.device

    max_m = min(K, U)

    # Outputs
    best_sum_rates = torch.full((B,), -float('inf'), device=device)
    best_masks = torch.zeros((B, U), device=device)

    # Precompute all subsets grouped by size
    subsets_by_m = {m: list(itertools.combinations(range(U), m)) for m in range(1, max_m + 1)}

    for m in range(1, max_m + 1):
        combos = subsets_by_m[m]
        C = len(combos)
        if C == 0:
            continue

        # Build action matrix for this m: [C, U]
        actions_all = torch.zeros((C, U), device=device)
        for i, subset in enumerate(combos):
            actions_all[i, list(subset)] = 1.0

        # Keep the best for this m so we can compare across m
        best_sum_rates_m = torch.full((B,), -float('inf'), device=device)
        best_masks_m = torch.zeros((B, U), device=device)

        # Process in chunks over combinations to avoid OOM
        for start in range(0, C, chunk_size):
            end = min(start + chunk_size, C)
            actions_chunk = actions_all[start:end]                          # [Cc, U]
            Cc = actions_chunk.shape[0]

            # Tile H over the combinations; tile actions over batch and flatten to (B*Cc)
            # H_rep: [B, Cc, U, Nt] -> [B*Cc, U, Nt]
            H_rep = H_batch.unsqueeze(1).expand(B, Cc, U, Nt).reshape(B * Cc, U, Nt)
            # actions_rep: [B, Cc, U] -> [B*Cc, U]
            actions_rep = actions_chunk.unsqueeze(0).expand(B, Cc, U).reshape(B * Cc, U)

            # Beamforming in parallel for all (b, combo) pairs
            if beamforming_method == 'zf':
                F = zf_beamforming(H_rep, actions_rep)         # expected [B*Cc, Nt, m] or similar
            else:
                F = rzf_beamforming(H_rep, actions_rep, noise_pwr)

            # Sum-rate for all pairs, then reshape to [B, Cc]
            sum_rate, _ = sum_rate_calculation(H_rep, actions_rep, F, noise_pwr)
            if isinstance(sum_rate, torch.Tensor):
                sum_rate = sum_rate.view(B, Cc)
            else:
                # In case it returns Python floats (unlikely), convert
                sum_rate = torch.tensor(sum_rate, device=device).view(B, Cc)
            # Enforce consistent dtype (avoid Float vs Double mismatches)
            sum_rate = sum_rate.to(dtype=torch.float32)

            # Argmax within the chunk for each b, compare to running best for m
            # current best per row
            chunk_best_vals, chunk_best_idx = sum_rate.max(dim=1)          # [B]
            # Align dtype with best_sum_rates_m destination
            chunk_best_vals = chunk_best_vals.to(best_sum_rates_m.dtype)
            improve_mask = chunk_best_vals > best_sum_rates_m              # [B]

            if improve_mask.any():
                # Update best values
                best_sum_rates_m[improve_mask] = chunk_best_vals[improve_mask]
                # Recover corresponding masks
                picked_idx = chunk_best_idx[improve_mask] + start          # global index in actions_all
                # Gather actions for those indices
                picked_actions = actions_all[picked_idx]                   # [#improve, U]
                # Assign into best_masks_m rows where improved
                best_masks_m[improve_mask] = picked_actions

        # Compare best across sizes m vs global best
        better_than_global = best_sum_rates_m > best_sum_rates
        if better_than_global.any():
            best_sum_rates[better_than_global] = best_sum_rates_m[better_than_global]
            best_masks[better_than_global] = best_masks_m[better_than_global]

    return best_masks, best_sum_rates


# def exhaustive_scheduling_batched(
#     H_batch: torch.Tensor,
#     K: int,
#     beamforming_method: str = 'zf',
#     ):
    
#     """
#     Exhaustively search user subsets (size 1..K) for each sample in a batch
#     and pick the per-sample subset that maximizes sum-rate.

#     Args
#     ----
#     H_batch : torch.Tensor
#         Channel tensor of shape [B, U, Nt] or [B, U, 1, Nt] (the latter will be squeezed).
#     K : int
#         Max number of users to schedule.
#     beamforming_method : str
#         'zf' or 'rzf'.

#     Returns
#     -------
#     best_masks : torch.Tensor
#         Tensor of shape [B, U] with 1's on the selected users for each sample.
#     best_sum_rates : torch.Tensor
#         Tensor of shape [B] with the best sum-rate achieved for each sample.
#     """
#     from baseline_methods.sched_bf_modules import zf_beamforming, rzf_beamforming, sum_rate_calculation

#     assert H_batch.dim() in (3, 4), "Expected H_batch with 3 or 4 dims: [B,U,Nt] or [B,U,1,Nt]"
#     if H_batch.dim() == 4:
#         # [B, U, 1, Nt] -> [B, U, Nt]
#         H_batch = H_batch.squeeze(2)

#     B, U, Nt = H_batch.shape
#     device = H_batch.device

#     # Precompute all subsets up to K once (they are shared across the batch)
#     max_m = min(K, U)
#     subsets_by_m = {m: list(itertools.combinations(range(U), m)) for m in range(1, max_m + 1)}

#     # Outputs
#     best_masks = torch.zeros((B, U), device=device)
#     best_sum_rates = torch.full((B,), -float('inf'), device=device)

#     # Loop per-sample (keeps behavior identical to your single-sample function)
#     for b in range(B):
#         H_b = H_batch[b:b+1, ...]  # shape [1, U, Nt] to match your existing APIs

#         best_sum_rate_b = -float('inf')
#         best_mask_b = None

#         for m in range(1, max_m + 1):
#             best_for_m = -float('inf')
#             best_mask_for_m = None

#             for subset in subsets_by_m[m]:
#                 actions = torch.zeros((1, U), device=device)
#                 actions[0, list(subset)] = 1.0

#                 # Beamforming
#                 if beamforming_method == 'zf':
#                     F = zf_beamforming(H_b, actions)
#                 else:
#                     F = rzf_beamforming(H_b, actions, noise_pwr)

#                 # Sum-rate (expecting a scalar for the single sample)
#                 sum_rate, _ = sum_rate_calculation(H_b, actions, F, noise_pwr)

#                 # If your sum_rate_calculation returns a tensor, convert to scalar
#                 if isinstance(sum_rate, torch.Tensor):
#                     # Assume shape [] or [1]; take item()
#                     sum_rate_val = sum_rate.squeeze().item()
#                 else:
#                     sum_rate_val = float(sum_rate)

#                 if sum_rate_val > best_for_m:
#                     best_for_m = sum_rate_val
#                     best_mask_for_m = actions.clone()

#             if best_for_m > best_sum_rate_b:
#                 best_sum_rate_b = best_for_m
#                 best_mask_b = best_mask_for_m

#         best_masks[b] = best_mask_b.squeeze(0)
#         best_sum_rates[b] = best_sum_rate_b

#     return best_masks, best_sum_rates


def zf_beamforming_batch(H_batch: torch.Tensor, actions_batch: torch.Tensor) -> torch.Tensor:
    """
    Batch ZF beamforming for multiple user selections.
    
    Args:
        H_batch: [batch_size, num_users, Nt]
        actions_batch: [batch_size, num_users]
    
    Returns:
        F_batch: [batch_size, Nt, K_eff] where K_eff is max scheduled users
    """
    batch_size, num_users, Nt = H_batch.shape
    device = H_batch.device
    
    # Find maximum number of scheduled users across batch
    K_eff = int(actions_batch.sum(dim=1).max().item())
    
    if K_eff == 0:
        return torch.zeros((batch_size, Nt, 1), device=device)
    
    F_batch = torch.zeros((batch_size, Nt, K_eff), device=device)
    
    
    for i in range(min(batch_size, actions_batch.shape[0])):
        # Get selected users for this batch element
        selected_users = (actions_batch[i] == 1).nonzero(as_tuple=True)[0]
        K_i = len(selected_users)
        
        if K_i == 0:
            continue
            
        # Get channel matrix for selected users
        H_selected = H_batch[i, selected_users, :]  # [K_i, Nt]
        
        # ZF beamforming: F = H^H (H H^H)^(-1)
        H_H = H_selected.conj().T  # [Nt, K_i]
        H_HH = torch.mm(H_selected, H_H)  # [K_i, K_i]
        
        try:
            H_HH_inv = torch.inverse(H_HH + 1e-6 * torch.eye(K_i, device=device))
            F_i = torch.mm(H_H, H_HH_inv)  # [Nt, K_i]
            
            # Normalize columns
            F_i = F_i / (torch.norm(F_i, dim=0, keepdim=True) + 1e-10)
            
            # Convert to real if complex
            if torch.is_complex(F_i):
                F_i = F_i.real
            
            F_batch[i, :, :K_i] = F_i
            
        except:
            # Fallback to pseudo-inverse
            F_i = torch.mm(H_H, torch.pinverse(H_HH))
            F_i = F_i / (torch.norm(F_i, dim=0, keepdim=True) + 1e-10)
            
            # Convert to real if complex
            if torch.is_complex(F_i):
                F_i = F_i.real
            
            F_batch[i, :, :K_i] = F_i
    
    return F_batch


def rzf_beamforming_batch(H_batch: torch.Tensor, actions_batch: torch.Tensor, noise_pwr: float = 1e-3) -> torch.Tensor:
    """
    Batch RZF beamforming for multiple user selections.
    """
    batch_size, num_users, Nt = H_batch.shape
    device = H_batch.device
    
    # Find maximum number of scheduled users across batch
    K_eff = int(actions_batch.sum(dim=1).max().item())
    
    if K_eff == 0:
        return torch.zeros((batch_size, Nt, 1), device=device)
    
    F_batch = torch.zeros((batch_size, Nt, K_eff), device=device)
    
    for i in range(min(batch_size, actions_batch.shape[0])):
        selected_users = (actions_batch[i] == 1).nonzero(as_tuple=True)[0]
        K_i = len(selected_users)
        
        if K_i == 0:
            continue
            
        H_selected = H_batch[i, selected_users, :]  # [K_i, Nt]
        
        # RZF beamforming: F = H^H (H H^H + αI)^(-1)
        H_H = H_selected.conj().T  # [Nt, K_i]
        H_HH = torch.mm(H_selected, H_H)  # [K_i, K_i]
        
        # Regularization parameter
        alpha = noise_pwr * K_i / Nt
        
        try:
            H_HH_reg = H_HH + alpha * torch.eye(K_i, device=device)
            H_HH_reg_inv = torch.inverse(H_HH_reg)
            F_i = torch.mm(H_H, H_HH_reg_inv)
            
            # Normalize columns
            F_i = F_i / (torch.norm(F_i, dim=0, keepdim=True) + 1e-10)
            
            # Convert to real if complex
            if torch.is_complex(F_i):
                F_i = F_i.real
            
            F_batch[i, :, :K_i] = F_i
            
        except:
            # Fallback to pseudo-inverse
            H_HH_reg = H_HH + alpha * torch.eye(K_i, device=device)
            F_i = torch.mm(H_H, torch.pinverse(H_HH_reg))
            F_i = F_i / (torch.norm(F_i, dim=0, keepdim=True) + 1e-10)
            
            # Convert to real if complex
            if torch.is_complex(F_i):
                F_i = F_i.real
            
            F_batch[i, :, :K_i] = F_i
    
    return F_batch


def sum_rate_calculation_batch(H_batch: torch.Tensor, actions_batch: torch.Tensor, 
                             F_batch: torch.Tensor, noise_pwr: float = 1e-3) -> torch.Tensor:
    """
    Batch sum-rate calculation for multiple user selections.
    
    Args:
        H_batch: [batch_size, num_users, Nt]
        actions_batch: [batch_size, num_users]
        F_batch: [batch_size, Nt, K_eff]
        noise_pwr: noise power
    
    Returns:
        sum_rates: [batch_size] - sum rates for each batch element
    """
    batch_size, num_users, Nt = H_batch.shape
    device = H_batch.device
    
    sum_rates = torch.zeros(batch_size, device=device)
    
    for i in range(min(batch_size, actions_batch.shape[0])):
        # Get selected users for this batch element
        selected_users = (actions_batch[i] == 1).nonzero(as_tuple=True)[0]
        K_i = len(selected_users)
        
        if K_i == 0:
            continue
            
        # Get channel and beamforming for selected users
        H_selected = H_batch[i, selected_users, :]  # [K_i, Nt]
        F_selected = F_batch[i, :, :K_i]  # [Nt, K_i]
        
        # Ensure compatible dtypes
        if torch.is_complex(H_selected) and not torch.is_complex(F_selected):
            F_selected = F_selected.to(H_selected.dtype)
        elif not torch.is_complex(H_selected) and torch.is_complex(F_selected):
            H_selected = H_selected.to(F_selected.dtype)
        
        # Calculate effective channels: G = H * F
        G = torch.mm(H_selected, F_selected)  # [K_i, K_i]
        
        # Calculate rates
        rates = torch.zeros(K_i, device=device)
        for k in range(K_i):
            # Signal power
            signal_power = torch.abs(G[k, k]) ** 2
            
            # Interference power
            interference_power = torch.sum(torch.abs(G[k, :]) ** 2) - signal_power
            
            # SINR
            sinr = signal_power / (interference_power + noise_pwr)
            
            # Rate
            rates[k] = torch.log2(1 + sinr)
        
        sum_rates[i] = torch.sum(rates)
    
    return sum_rates
