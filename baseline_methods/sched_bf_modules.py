import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from config import *
from typing import Tuple


def random_scheduling(actions: torch.Tensor, K: int) -> torch.Tensor:
    """
    Randomly selects up to K of the '1's in each row (batch element).
    If #1's <= K, keep them all; if >K, randomly choose K of them.
    """
    final_actions = actions.clone()
    batch_size, num_users = actions.shape

    for i in range(batch_size):
        count_ones = int(final_actions[i].sum().item())
        if count_ones > K:
            one_indices = (final_actions[i] == 1).nonzero(as_tuple=True)[0]
            chosen = torch.randperm(count_ones)[:K]
            chosen_indices = one_indices[chosen]
            final_actions[i] = 0
            final_actions[i, chosen_indices] = 1

    return final_actions


def topK_scheduling(actions: torch.Tensor, H_batch: torch.Tensor, K: int) -> torch.Tensor:
    """
    Vectorized selection of top K users based on channel gain from nominated candidates.

    Parameters:
    ----------
    actions : torch.Tensor
        Binary tensor of shape (batch_size, num_users), where '1' indicates a candidate user.
    H_batch : torch.Tensor
        Complex channel tensor of shape (batch_size, num_users, 1, Nt).
    K : int
        Maximum number of users to select per batch instance.

    Returns:
    -------
    final_actions : torch.Tensor
        Binary tensor of shape (batch_size, num_users) with selected users marked as '1'.
    """
    batch_size, num_users = actions.shape

    # Ensure K does not exceed num_users
    K = min(K, num_users)
    
    # Compute channel gains: squared Frobenius norm of each user's channel vector
    channel_gains = torch.sum(torch.abs(H_batch)**2, dim=(2, 3))  # Shape: [batch_size, num_users]

    # Mask channel gains with actions (set non-candidates to -inf for proper topk selection)
    masked_gains = channel_gains.clone()
    masked_gains[actions == 0] = -float('inf')  # Non-candidates have -inf gain

    # Use torch.topk to select top K users based on channel gains
    topk_gains, topk_indices = torch.topk(masked_gains, K, dim=1, largest=True, sorted=False)  # Shape: [batch_size, K]

    # Create a mask for top K selections
    final_actions = torch.zeros_like(actions, dtype=torch.float32)  # Initialize to zero

    # Scatter 1s into final_actions at the top K indices
    final_actions.scatter_(1, topk_indices, 1.0)

    # Handle cases where the number of candidates is less than K
    # Users with -inf gains were selected but should not be marked as 1
    # Create a mask where masked_gains > -inf
    candidates_mask = masked_gains > -float('inf')  # Shape: [batch_size, num_users]
    final_actions = final_actions * candidates_mask.float().to(actions.device)

    return final_actions



def exhaustive_scheduling(H_sample: torch.Tensor, K: int, beamforming_method: str = 'rzf',
                          noise_pwr: float = 1e-3):
    """
    Exhaustively search all possible user subsets (1..num_users) to find the one 
    giving the best sum-rate. Evaluate all sizes up to K and keep the global best.
    """
    from baseline_methods.sched_bf_modules import zf_beamforming, rzf_beamforming, sum_rate_calculation

    # Prepare channel
    if H_sample.dim() == 4:
        H_sample = H_sample.squeeze(2)   # [1, num_users, Nt]
    num_users = H_sample.shape[1]

    # To store results
    best_sum_rate = -float('inf')
    best_mask = None

    # Iterate over group sizes 1..K
    for m in range(1, min(K, num_users) + 1):
        best_for_m = -float('inf')
        best_mask_for_m = None

        # Generate all combinations of m users
        for subset in itertools.combinations(range(num_users), m):
            actions = torch.zeros((1, num_users), device=H_sample.device)
            actions[0, list(subset)] = 1

            # Beamforming
            if beamforming_method == 'zf':
                F = zf_beamforming(H_sample, actions)
            else:
                F = rzf_beamforming(H_sample, actions, noise_pwr)

            # Sum-rate
            sum_rate, _ = sum_rate_calculation(H_sample, actions, F, noise_pwr)

            if sum_rate > best_for_m:
                best_for_m = sum_rate
                best_mask_for_m = actions.clone()

        # Keep global best (no early stop)
        if best_for_m > best_sum_rate:
            best_sum_rate = best_for_m
            best_mask = best_mask_for_m

    return best_mask, best_sum_rate


def pf_scheduling(H_batch: torch.Tensor,
                              actions: torch.Tensor,
                              weights: torch.Tensor,
                              K: int) -> torch.Tensor:
    """
    PF-style scheduling among nominated users, returning final 0/1 mask (batch_size, num_users).

    Inputs:
      H_batch:   (batch_size, num_users, Nt)
        Channel data for each user in the batch.
      actions:   (batch_size, num_users) => 0/1 from self-nomination 
      weights: (batch_size, num_users) => user average rates (for PF weight=1/avg_rates) but randomly generated and fixed
      K:         integer, max # of final scheduled users per batch row

    Returns:
      final_actions: (batch_size, num_users) in {0,1}
        where each row has up to K=1's among the nominated set,
        picked by PF scoring if #nominated> K,
        or all nominated if #nominated <= K.

    Approx Scoring:
      - feasible_rate_k ~ log2(1 + ||h_k||^2),
      - weight_k = 1 / avg_rates[k],
      - total_score = feasible_rate_k * weight_k.
    """
    device = H_batch.device
    batch_size, num_users, _, Nt = H_batch.shape

    # 1) feasible_rates = log2(1 + ||H_batch||^2) => shape (b, n)
    norm_sq = torch.linalg.norm(H_batch, dim=(2,3))**2  # => shape [b, n]
    feasible_rates = torch.log2(1.0 + norm_sq/num_users)

    # 2) weights => 1 / avg_rates => shape [b, n]
    # w = 1.0 / torch.clamp(avg_rates, min=1e-9)

    # 3) raw_score => feasible_rates * w => shape (b, n)
    score = feasible_rates * weights

    # 4) set non-nominated to -inf so they're never picked
    #    actions==0 => not nominated
    neg_inf = torch.tensor(float('-inf'), device=device, dtype=score.dtype)
    score = torch.where(actions > 0.5, score, neg_inf)

    # We'll build final_actions by row, handling corner case #nominated <= K
    final_actions = torch.zeros_like(actions)

    # 5) number of nominated per row
    nominated_count = actions.sum(dim=1)  # shape => [b]

    for i in range(batch_size):
        c_ones = int(nominated_count[i].item())
        if c_ones <= K:
            # If #nominated <= K, schedule all nominated
            final_actions[i] = actions[i]
        else:
            # topK among this row i
            row_score = score[i]       # shape => (n,)
            # get top K
            val, idx = torch.topk(row_score, k=K, dim=0)
            row_final = torch.zeros(num_users, device=device, dtype=actions.dtype)
            row_final[idx] = 1.0
            final_actions[i] = row_final

    return final_actions


def mrt_beamforming(H_batch: torch.Tensor, final_actions: torch.Tensor) -> torch.Tensor:
    """
    Performs Maximum Ratio Transmission (MRT) on the selected users (final_actions).
    Returns a normalized beamforming matrix F_MRT_norm for each sample.

    Parameters:
    ----------
    H_batch : torch.Tensor
        Complex channel tensor with shape [batch_size, num_users, 1, Nt] or [batch_size, num_users, Nt].
        Represents the channel from the base station to each user.
    final_actions : torch.Tensor
        Binary tensor of shape [batch_size, num_users], where '1' indicates a selected user.

    Returns:
    -------
    F_MRT_norm : torch.Tensor
        Normalized MRT beamforming matrix with shape [batch_size, Nt, num_users].
    """
    device = H_batch.device
    epsilon = 1e-7

    # If H_batch has a singleton dimension at dim=2, squeeze it out
    if H_batch.dim() == 4:
        H_selected = H_batch.squeeze(2) * final_actions.unsqueeze(-1)  # Shape: [batch_size, num_users, Nt]
    else:
        # If H_batch is already [batch_size, num_users, Nt]
        H_selected = H_batch * final_actions.unsqueeze(-1)  # Shape: [batch_size, num_users, Nt]

    # Compute the conjugate of the selected channels
    H_conj = H_selected.conj()  # Shape: [batch_size, num_users, Nt]

    # # Compute the norm of each channel vector
    # H_norm = torch.norm(H_selected, dim=2, keepdim=True) + epsilon  # Shape: [batch_size, num_users, 1]

    # Compute MRT beamforming vectors: f_i = h_i^* / ||h_i||
    F_MRT = H_conj 
    # / H_norm  # Shape: [batch_size, num_users, Nt]

    # Transpose to get shape [batch_size, Nt, num_users]
    F_MRT = F_MRT.transpose(1, 2)  # Shape: [batch_size, Nt, num_users]

    # Normalize the beamforming matrix per user to have unit norm
    # Compute the norm of each beamforming vector
    F_norm = torch.norm(F_MRT, dim=1, keepdim=True) + epsilon  # Shape: [batch_size, 1, num_users]

    # Normalize F_MRT
    F_MRT_norm = F_MRT / F_norm  # Shape: [batch_size, Nt, num_users]
    F_MRT_norm_entire = F_MRT_norm / torch.norm(F_MRT_norm , dim=(1,2), keepdim=True) + epsilon
    
    return F_MRT_norm_entire



def zf_beamforming(H_batch: torch.Tensor, final_actions: torch.Tensor) -> torch.Tensor:
    """
    Performs Zero-Forcing on the selected users (final_actions) but does NOT compute sum-rate.
    Returns a normalized beamforming matrix F_ZF_norm for each sample.

    H_batch: shape [batch_size, num_users, 1, Nt] or [batch_size, num_users, Nt]
    final_actions: shape [batch_size, num_users], 0/1 for each user

    Returns:
      F_ZF_norm: shape [batch_size, Nt, num_users], the ZF beamforming matrix (normalized).
    """
    device = H_batch.device
    epsilon = 1e-7

    # If H_batch is [batch, num_users, 1, Nt], squeeze out the 3rd dim
    if H_batch.dim() == 4:
        H_selected = H_batch.squeeze(2) * final_actions.unsqueeze(-1)  # shape: [batch, num_users, Nt]
    else:
        # If it's already [batch, num_users, Nt]
        H_selected = H_batch * final_actions.unsqueeze(-1)
    
    # if not isinstance(final_actions, torch.Tensor):
    #     final_actions = torch.from_numpy(final_actions).to(device)


    # [batch, num_users, Nt]
    H_herm = H_selected.transpose(1, 2).conj()  # [batch, Nt, num_users]
    
    batch_size, num_users, Nt = H_selected.shape
    # H H^H + eps I
    Hmul = torch.matmul(H_selected, H_herm) \
         + epsilon * torch.eye(num_users, dtype=torch.cfloat, device=device).unsqueeze(0)
    Hmul_inv = torch.linalg.inv(Hmul)  # [batch, num_users, num_users]

    # F_ZF = H^H * (H H^H)^{-1}
    F_ZF = torch.matmul(H_herm, Hmul_inv)  # [batch, Nt, num_users]

    # Normalize columns so that each user has unit norm or the entire matrix has unit norm?
    # Here, let's do user-wise column normalization:
    per_ue_norm = torch.norm(F_ZF, p=2, dim=1, keepdim=True) + epsilon  # shape [batch, 1, num_users]
    F_ZF_norm = F_ZF / per_ue_norm
    norms = torch.norm(F_ZF_norm, dim=(1, 2), keepdim=True) + epsilon
    # Normalize F_ZF to have unit norm for each batch element
    F_ZF_norm = F_ZF_norm / norms #(norms + epsilon)

    return F_ZF_norm

### RZF
def rzf_beamforming(H_batch: torch.Tensor, final_actions: torch.Tensor, noise_power: float = 1.0,) -> torch.Tensor:
    """
    Regularized-ZF / MMSE beamforming (per-batch alpha).

    alpha_b  =  K_active[b] / SNR_lin,   where
      K_active[b] = sum_k final_actions[b,k]  (≥1)  ← only scheduled UEs
      SNR_lin     = 10**(snr_db / 10)

    Parameters
    ----------
    H_batch : complex torch.Tensor
        [B, K_tot, 1, Nt] or [B, K_tot, Nt] channel.
    final_actions : torch.Tensor
        [B, K_tot] {0,1} mask. 1 = UE scheduled.
    snr_db : float
        Per-stream SNR in dB used to set alpha (default 10 dB).
    noise_power : float
        Kept for completeness; not used in the default alpha rule.

    Returns
    -------
    F_RZF_norm : torch.Tensor
        [B, Nt, K_tot] power-normalised RZF precoder
        (columns for unscheduled users will be ≈0).
    """
    device = H_batch.device
    eps    = 1e-7

    if not isinstance(final_actions, torch.Tensor):
        final_actions = torch.from_numpy(final_actions).to(device)

    # ---- Reshape to [B, K_tot, Nt] and zero out unscheduled UEs ----
    if H_batch.dim() == 4:
        H_sel = H_batch.squeeze(2) * final_actions.unsqueeze(-1)
    else:
        H_sel = H_batch * final_actions.unsqueeze(-1)             # [B,K_tot,Nt]

    H_H   = H_sel.transpose(1, 2).conj()                          # [B,Nt,K_tot]
    B, K_tot, Nt = H_sel.shape

    # ---- Per-batch α based on *active* users ----
    K_active = final_actions.sum(dim=1, keepdim=True).clamp(min=1)  # [B,1]
    alpha    = (K_active * noise_power).view(B, 1, 1)                  # [B,1,1]

    # ---- (H H^H + α I)^{-1} ----
    IdK = torch.eye(K_tot, dtype=torch.cfloat, device=device).unsqueeze(0)  # [1,K,K]
    gram = torch.matmul(H_sel, H_H) + alpha * IdK                           # [B,K,K]
    gram_inv = torch.linalg.inv(gram)                                       # [B,K,K]

    # ---- Precoder F = H^H (H H^H + αI)^{-1} ----
    F_rzf = torch.matmul(H_H, gram_inv)                                     # [B,Nt,K_tot]

    # ---- Column-wise normalisation ----
    col_norm  = torch.norm(F_rzf, p=2, dim=1, keepdim=True) + eps
    F_rzf     = F_rzf / col_norm                                            # unit columns
    batch_pow = torch.norm(F_rzf, dim=(1, 2), keepdim=True) + eps
    F_RZF_norm = F_rzf / batch_pow                                          # unit total power

    return F_RZF_norm



def zf_beamforming_alt(H_batch: torch.Tensor, final_actions: torch.Tensor) -> torch.Tensor:
    """
    Performs Zero-Forcing on the selected users (final_actions) but does NOT compute sum-rate.
    Returns a normalized beamforming matrix F_ZF_norm for each sample.

    Using the pseudo-inverse form: F_ZF = (H^H H)^{-1} * H^H.

    Args:
        H_batch: shape [batch_size, num_users, Nt]  (or [batch_size, num_users, 1, Nt], see code)
        final_actions: shape [batch_size, num_users], 0/1 for each user

    Returns:
        F_ZF_norm: shape [batch_size, Nt, num_users], the ZF beamforming matrix (normalized).
    """
    device = H_batch.device
    epsilon = 1e-7

    # If H_batch is [batch, num_users, 1, Nt], squeeze out the 3rd dim
    if H_batch.dim() == 4:
        # shape [batch, num_users, Nt]
        H_selected = H_batch.squeeze(2) * final_actions.unsqueeze(-1)
    else:
        # Already [batch, num_users, Nt]
        H_selected = H_batch * final_actions.unsqueeze(-1)

    # H_herm = [batch, Nt, num_users]
    H_herm = H_selected.transpose(1, 2).conj()

    # (H^H H): [batch, Nt, Nt]
    batch_size, num_users, Nt = H_selected.shape
    I_Nt = torch.eye(Nt, dtype=torch.cfloat, device=device).unsqueeze(0)  # [1, Nt, Nt]

    # Hmul = H^H H + eps*I
    Hmul = torch.matmul(H_herm, H_selected) + epsilon * I_Nt
    # Invert [batch, Nt, Nt]
    Hmul_inv = torch.linalg.inv(Hmul)

    # F_ZF = (H^H H)^-1 H^H => [batch, Nt, num_users]
    F_ZF = torch.matmul(Hmul_inv, H_herm)

    # -- Normalization ---------------------------------------------------
    # 1) Per-user column normalization
    #    Each column (i.e., each user) has norm = sqrt(sum(|F_ZF[:, user]|^2)).
    per_ue_norm = torch.norm(F_ZF, p=2, dim=1, keepdim=True) + epsilon  # [batch, 1, num_users]
    F_ZF_norm = F_ZF / per_ue_norm

    # 2) (Optional) normalize the entire F_ZF to unit Frobenius norm per batch
    #    This step depends on your convention. If you want total power = 1 for each batch:
    total_norm = torch.norm(F_ZF_norm, dim=(1, 2), keepdim=True) + epsilon  # [batch, 1, 1]
    F_ZF_norm = F_ZF_norm / total_norm

    return F_ZF_norm


def sum_rate_calculation(
    H_batch: torch.Tensor,
    final_actions: torch.Tensor,
    F_ZF_norm: torch.Tensor,
    noise_pwr: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given final_actions (which subset of users are scheduled),
    the normalized ZF beamforming matrix F_ZF_norm,
    and user-specific weights, compute the *weighted* sum-rate.
    
    H_batch:    [batch_size, num_users, (1,) Nt]
    final_actions: [batch_size, num_users] (0 or 1)
    F_ZF_norm:  [batch_size, Nt, num_users]
    noise_pwr:  scalar noise power
    
    Returns:
      (weighted_sum_rate, avg_weighted_sum_rate)
       where weighted_sum_rate is shape [batch_size],
       and avg_weighted_sum_rate is a scalar average.
    """
    device = H_batch.device
    epsilon = 1e-7

    # If H_batch is [batch, num_users, 1, Nt], remove the extra dim
    if H_batch.dim() == 4:
        H_selected = H_batch.squeeze(2) * final_actions.unsqueeze(-1)  # [batch, num_users, Nt]
    else:
        H_selected = H_batch * final_actions.unsqueeze(-1)

    # Gains => |H * F|^2
    chg_mat_vec = torch.square(torch.abs(torch.matmul(H_selected, F_ZF_norm)))  # [batch, num_users, num_users]

    # Diagonal => signal power
    numerator = torch.diagonal(chg_mat_vec, dim1=1, dim2=2)  # [batch, num_users]
    # Row sum => total received power
    row_sums = chg_mat_vec.sum(dim=2)  # [batch, num_users]
    denominator = row_sums - numerator + noise_pwr

    # rate_tmp = log2(1 + numerator/denominator)
    rate_tmp = torch.log1p(numerator / denominator)
    rate_vec = rate_tmp / torch.log(torch.tensor(2.0, device=device))  # shape [batch, num_users]

    # If your final_actions is already 1 or 0 for scheduled vs. not, that will mask out
    weighted_rate = rate_vec * final_actions
    # weighted_rate = rate_vec * final_actions

    # Weighted sum-rate across users
    weighted_sum_rate = weighted_rate.sum(dim=1)  # [batch_size]
    avg_weighted_sum_rate = weighted_sum_rate.mean()

    return weighted_sum_rate, avg_weighted_sum_rate



def weighted_sum_rate_calculation(
    H_batch: torch.Tensor,
    final_actions: torch.Tensor,
    F_ZF_norm: torch.Tensor,
    noise_pwr: float,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given final_actions (which subset of users are scheduled),
    the normalized ZF beamforming matrix F_ZF_norm,
    and user-specific weights, compute the *weighted* sum-rate.
    
    H_batch:    [batch_size, num_users, (1,) Nt]
    final_actions: [batch_size, num_users] (0 or 1)
    F_ZF_norm:  [batch_size, Nt, num_users]
    noise_pwr:  scalar noise power
    weights:    [batch_size, num_users] user-specific weights
    
    Returns:
      (weighted_sum_rate, avg_weighted_sum_rate)
       where weighted_sum_rate is shape [batch_size],
       and avg_weighted_sum_rate is a scalar average.
    """
    device = H_batch.device
    epsilon = 1e-7

    # If H_batch is [batch, num_users, 1, Nt], remove the extra dim
    if H_batch.dim() == 4:
        H_selected = H_batch.squeeze(2) * final_actions.unsqueeze(-1)  # [batch, num_users, Nt]
    else:
        H_selected = H_batch * final_actions.unsqueeze(-1)

    # Gains => |H * F|^2
    chg_mat_vec = torch.square(torch.abs(torch.matmul(H_selected, F_ZF_norm)))  # [batch, num_users, num_users]

    # Diagonal => signal power
    numerator = torch.diagonal(chg_mat_vec, dim1=1, dim2=2)  # [batch, num_users]
    # Row sum => total received power
    row_sums = chg_mat_vec.sum(dim=2)  # [batch, num_users]
    denominator = row_sums - numerator + noise_pwr

    # rate_tmp = log2(1 + numerator/denominator)
    rate_tmp = torch.log1p(numerator / denominator)
    rate_vec = rate_tmp / torch.log(torch.tensor(2.0, device=device))  # shape [batch, num_users]

    # Now apply the weights *and* final_actions (if you only want scheduled users counted)
    # weighted_rate = weights * rate_vec * final_actions
    # If your final_actions is already 1 or 0 for scheduled vs. not, that will mask out
    weighted_rate = weights * rate_vec * final_actions
    # weighted_rate = rate_vec * final_actions

    # Weighted sum-rate across users
    weighted_sum_rate = weighted_rate.sum(dim=1)  # [batch_size]
    avg_weighted_sum_rate = weighted_sum_rate.mean()

    return weighted_sum_rate, avg_weighted_sum_rate


def sus_scheduling(actions: torch.Tensor,
                   H_batch: torch.Tensor,  # (B, U, 1, Nt) complex
                   K: int,
                   alpha: float) -> torch.Tensor:
    device = H_batch.device
    eps = 1e-10

    # shapes
    B, U, _, Nt = H_batch.shape
    H = H_batch.squeeze(2)  # (B, U, Nt) complex
    cand = actions.bool().clone()  # (B, U)
    # import pdb; pdb.set_trace()
    final_actions = torch.zeros_like(actions, dtype=torch.float32, device=device)
    G = torch.zeros(B, 0, Nt, dtype=H.dtype, device=device)  # store orth residuals g_i

    # Keep original H for pruning correlations
    H_orig = H

    for i in range(K):
        # If nothing left, stop
        any_cand = cand.any(dim=1)  # (B,)
        if not any_cand.any():
            break

        # Compute residuals w.r.t. current G
        if G.size(1) == 0:
            R = H  # (B, U, Nt)
        else:
            # inner products: <g, h>
            ip = torch.einsum("buk,bvk->buv", H, G.conj())            # (B, U, i)
            gnorm2 = (G.abs()**2).sum(dim=2).clamp_min(eps)           # (B, i)
            coeff = ip / gnorm2.unsqueeze(1)                          # (B, U, i)
            proj = torch.einsum("bui,bik->buk", coeff, G.conj())      # (B, U, Nt)
            R = H - proj                                              # (B, U, Nt)

        # pick argmax residual norm among candidates
        rnorm = R.norm(dim=2)                                         # (B, U)
        rnorm_masked = rnorm.masked_fill(~cand, -1.0)
        best_vals, best_idx = rnorm_masked.max(dim=1)                 # (B,)

        # batches with no valid candidate this round
        valid = best_vals > 0
        if not valid.any():
            break

        # new direction is the residual vector (orth component!)
        bidx = torch.arange(B, device=device)
        g_new = torch.zeros(B, Nt, dtype=H.dtype, device=device)
        g_new[valid] = R[valid, best_idx[valid]]
        # normalize optional (not required for SUS tests, but stable)
        # g_new = g_new / g_new.norm(dim=1, keepdim=True).clamp_min(eps)

        # append to G
        G = torch.cat([G, g_new.unsqueeze(1)], dim=1)                 # (B, i+1, Nt)

        # mark selected in output and remove from candidates
        final_actions[valid, best_idx[valid]] = 1.0
        cand[bidx[valid], best_idx[valid]] = False

        # prune by semi-orthogonality: corr(g_new, h_j) < alpha
        # use original H for pruning, like the numpy version
        g = g_new.unsqueeze(1)                                        # (B, 1, Nt)
        num = (H_orig.conj() * g).sum(dim=2).abs()                    # (B, U)
        den = (H_orig.norm(dim=2) * g_new.norm(dim=1, keepdim=True)).clamp_min(eps)
        corr = num / den                                              # (B, U)

        # keep only those still candidates, not selected, and corr < alpha
        keep = (corr < alpha) & cand
        cand = keep

    return final_actions



