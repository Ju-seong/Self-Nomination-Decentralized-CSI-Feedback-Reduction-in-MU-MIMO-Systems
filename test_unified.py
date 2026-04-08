import os
import torch
import numpy as np
import scipy.io as spio
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from loaders import CommunicationDataset
from config import *
from baseline_methods.sched_bf_modules import *

# CUDA device will be set via command line argument


def get_model_class(method, input_type):
    """
    Get the appropriate model class based on method and input type.
    
    Args:
        method (str): 'reinforce' or 'directgrad'
        input_type (str): 'full' or 'chg_input'
    
    Returns:
        Model class
    """
    if method == 'reinforce' and input_type == 'full':
        from learning_modules.reinforce_full_input import Modules
        return Modules
    elif method == 'reinforce' and input_type == 'chg_input':
        from learning_modules.reinforce_channel_gain_input import Modules_chg as Modules
        return Modules
    elif method == 'directgrad' and input_type == 'full':
        from learning_modules.direct_gradient_full_input import Modules
        return Modules
    elif method == 'directgrad' and input_type == 'chg_input':
        from learning_modules.direct_gradient_channel_gain_input import Modules
        return Modules
    else:
        raise ValueError(f"Invalid combination: method={method}, input_type={input_type}")


def is_baseline_method(scheduling, beamforming):
    """
    Determine if the combination of scheduling and beamforming methods should use baseline methods.
    
    Args:
        scheduling (str): Scheduling method
        beamforming (str): Beamforming method
    
    Returns:
        bool: True if baseline method should be used, False for deep learning
    """
    # Define baseline method combinations
    baseline_combinations = [
        # Pure baseline methods (no deep learning nomination)
        ('random', 'zf'),
        ('random', 'rzf'),
        ('exhaustive', 'zf'),
        ('exhaustive', 'rzf'),
        ('pf', 'zf'),
        ('pf', 'rzf'),
        ('sus', 'zf'),
        ('sus', 'rzf'),
        # TopK with specific beamforming (traditional greedy)
        ('greedy', 'zf'),
        ('greedy', 'rzf'),
    ]
    
    return (scheduling, beamforming) in baseline_combinations


class BaselineMethod:
    """
    Baseline method wrapper that mimics the interface of deep learning models.
    """
    def __init__(self, scheduling_method='greedy', beamforming_method='zf'):
        self.scheduling_method = scheduling_method
        self.beamforming_method = beamforming_method
        
    def eval(self):
        """Set to evaluation mode (no-op for baseline)."""
        pass
        
    def parameters(self):
        """Return empty parameters iterator for baseline methods."""
        return iter([])
        
    def to(self, device):
        """Move to device (no-op for baseline)."""
        return self
        
    def load_state_dict(self, state_dict):
        """Load state dict (no-op for baseline)."""
        pass


def initialize_device():
    """Initialize and return the appropriate device for computation."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(dataset, num_train, num_val, num_test, channel_mode="UMi_UPA"):
    """Create and return a DataLoader for the test dataset."""
    # For 3GPP scenarios (UMi_UPA, UMi_ULA, Berlin_UPA), use more conservative test data selection to avoid overlap with training
    if channel_mode in ["UMi_UPA", "UMi_ULA", "Berlin_UPA"]:
        # Start test data from a larger offset to ensure no overlap with training data
        test_start = num_train + num_val 
        test_indices = range(test_start, test_start + num_test)
    else:
        # For RF, use standard splitting
        test_indices = range(num_train + num_val, num_train + num_val + num_test)
    
    test_subset = Subset(dataset, test_indices)
    # return DataLoader(test_subset, batch_size=num_test, shuffle=False)
    return DataLoader(test_subset, batch_size=1, shuffle=False)


def load_model(model_path, method, input_type, scheduling_method='greedy', beamforming_method='zf', device=None):
    """
    Load model weights and return the initialized model or baseline method.
    
    Args:
        model_path (str): Path to model file (ignored for baseline methods)
        method (str): Training method ('reinforce', 'directgrad', or 'baseline')
        input_type (str): Input type ('full', 'chg_input', ignored for baseline)
        scheduling_method (str): Scheduling method
        beamforming_method (str): Beamforming method
        device: PyTorch device
    
    Returns:
        Model or BaselineMethod instance
    """
    if device is None:
        device = initialize_device()
    
    # Use baseline wrapper only when explicitly requested via method
    if method == 'baseline':
        print(f"Using baseline method: scheduling={scheduling_method}, beamforming={beamforming_method}")
        return BaselineMethod(scheduling_method=scheduling_method, beamforming_method=beamforming_method)
    
    # Load deep learning model
    ModelClass = get_model_class(method, input_type)
    model = ModelClass(scheduling_method=scheduling_method, beamforming_method=beamforming_method).to(device)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded deep learning model from: {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}, using untrained model")
    
    model.eval()  # Set the model to evaluation mode
    return model


def self_nomination_step(model, H_sample, method='reinforce', input_type='full'):
    """
    Self-nomination by proposed deep learning models only.
    H_sample: shape (num_users, 1, Nt) or (num_users, Nt) in NumPy or torch
    
    Returns:
      nominated_mask: a 0/1 array of shape (num_users,) indicating self-nomination.
    """
    # Handle deep learning models
    # Convert to torch
    if not torch.is_tensor(H_sample):
        H_torch = torch.from_numpy(H_sample)
    else:
        H_torch = H_sample

    device = next(model.parameters()).device
    H_torch = H_torch.squeeze(2).to(device)  # shape => [1, num_users, 1, Nt]
    
    with torch.no_grad():
        if hasattr(model, 'deep_sched'):
            # For models with deep_sched architecture
            if input_type == 'chg_input':
                # Channel gain input: one scalar per UE (||h||)
                H_sqz = H_torch if H_torch.dim() == 3 else H_torch.squeeze(2)
                chg = torch.norm(H_sqz, dim=2)
                chg_input = chg.view(-1, 1).float()
                logits = model.forward(chg_input) * model.sharpness
            else:
                # Full channel matrix input
                H_real = H_torch.real
                H_imag = H_torch.imag
                b_size, num_users = H_torch.shape[0], H_torch.shape[1]
                
                H_combined = torch.cat((H_real, H_imag), dim=2)
                H_input = H_combined.view(-1, 2, Nt).float()
                logits = model.forward(H_input) * model.sharpness
            
            logits = logits.view(H_torch.shape[0], H_torch.shape[1])
            
            # Sigmoid => probabilities
            prob = torch.sigmoid(logits)
            
            if method == 'reinforce':
                # Bernoulli sampling for REINFORCE
                nominated_mask_torch = torch.bernoulli(prob)
            else:
                # Threshold for direct gradient
                nominated_mask_torch = (prob > 0.5).float()
        else:
            # Fallback for other model types
            _, _, nominated_mask_torch = model.train_module(H_torch.unsqueeze(2), training=False)

    nominated_mask = nominated_mask_torch.cpu().numpy()
    nominated_mask = (nominated_mask > 0.5).astype(np.float32)
    return nominated_mask


def compute_orthogonality_defect(H_sample, nominated_mask):
    """
    Compute the orthogonality defect for the nominated channels.
    
    Args:
        H_sample: torch.Tensor, shape [batch_size, num_users, 1, Nt] or [batch_size, num_users, Nt]
        nominated_mask: numpy.array or torch.Tensor, shape [batch_size, num_users]
        
    Returns:
        orthogonality_defect: numpy.array, shape [batch_size]
    """
    epsilon = 1e-10  # For numerical stability
    
    if isinstance(nominated_mask, np.ndarray):
        nominated_mask = torch.from_numpy(nominated_mask).to(H_sample.device)
        
    # Handle dimension, squeeze if necessary
    if H_sample.dim() == 4:
        H_sample = H_sample.squeeze(2)  # [batch_size, num_users, Nt]

    batch_size, num_users, Nt = H_sample.shape

    defects = []
    for b in range(batch_size):
        nominated_indices = torch.where(nominated_mask[b] > 0.5)[0]
        if len(nominated_indices) == 0:
            defects.append(np.inf)
            continue
            
        H_selected = H_sample[b, nominated_indices, :]  # shape: [M, Nt]
        M_selected = H_selected.shape[0]

        # Compute numerator: product of squared norms
        squared_norms = torch.norm(H_selected, dim=1) ** 2  # [M]
        numerator = torch.prod(squared_norms) + epsilon

        # Compute denominator: determinant of H H^H
        HH_H = H_selected @ H_selected.conj().T  # [M, M]
        eigvals = torch.linalg.eigvalsh(HH_H)
        nonzero = eigvals[eigvals > 1e-8]
        if len(nonzero) == 0:
            defects.append(np.inf)
            continue
        det_HH_H = torch.prod(nonzero)  # Ensure positive and stable

        orth_defect = numerator / det_HH_H
        defects.append(orth_defect.cpu().numpy())
        
    return np.array(defects)  # [batch_size]


def compute_condition_number(H_sample, nominated_mask):
    """
    Compute the condition number for the nominated channels.
    
    Args:
        H_sample: torch.Tensor, shape [batch_size, num_users, 1, Nt] or [batch_size, num_users, Nt]
        nominated_mask: numpy.array or torch.Tensor, shape [batch_size, num_users]
        
    Returns:
        condition_numbers: numpy.array, shape [batch_size]
    """
    if isinstance(nominated_mask, np.ndarray):
        nominated_mask = torch.from_numpy(nominated_mask).to(H_sample.device)
        
    # Handle dimension, squeeze if necessary
    if H_sample.dim() == 4:
        H_sample = H_sample.squeeze(2)  # [batch_size, num_users, Nt]

    batch_size, num_users, Nt = H_sample.shape

    condition_numbers = []
    for b in range(batch_size):
        nominated_indices = torch.where(nominated_mask[b] > 0.5)[0]
        if len(nominated_indices) == 0:
            condition_numbers.append(np.inf)
            continue
            
        H_selected = H_sample[b, nominated_indices, :]  # shape: [M, Nt]
        
        # Compute condition number
        HH_H = H_selected @ H_selected.conj().T  # [M, M]
        eigvals = torch.linalg.eigvalsh(HH_H)
        nonzero = eigvals[eigvals > 1e-8]
        if len(nonzero) == 0:
            condition_numbers.append(np.inf)
            continue
            
        condition_num = torch.max(nonzero) / torch.min(nonzero)
        condition_numbers.append(condition_num.cpu().numpy())
        
    return np.array(condition_numbers)  # [batch_size]


def build_nomination_mask(
    H_sample,
    model,
    method,
    input_type,
    device,
    nomination_mode="model",
    random_fb_prob=0.5,
):
    """Construct the nomination mask for the requested evaluation mode."""
    num_users_local = H_sample.shape[1]

    if nomination_mode == "all_fb":
        nominated_mask_torch = torch.ones(1, num_users_local, device=device)
    elif nomination_mode == "random_fb":
        probs = torch.full((1, num_users_local), random_fb_prob, device=device)
        nominated_mask_torch = torch.bernoulli(probs)
    elif nomination_mode == "model":
        if method not in ["reinforce", "directgrad"]:
            raise ValueError(f"Model-based nomination is not available for method={method}")
        nominated_mask = self_nomination_step(model, H_sample, method, input_type)
        nominated_mask_torch = torch.from_numpy(nominated_mask).to(device)
    else:
        raise ValueError(f"Invalid nomination_mode: {nomination_mode}")

    nominated_mask = nominated_mask_torch.cpu().numpy().astype(np.float32)
    return nominated_mask, nominated_mask_torch


def evaluate_model(
    model,
    test_loader,
    method,
    device,
    input_type='full',
    noise_power=None,
    nomination_mode="model",
    random_fb_prob=0.5,
):
    """
    Evaluate the model on test data and compute various metrics.
    """
    if noise_power is None:
        noise_power = noise_pwr

    model.eval()
    
    all_sum_rates = []
    all_nominated_counts = []
    all_scheduled_counts = []
    all_orthogonality_defects = []
    all_condition_numbers = []
    
    print(f"Evaluating {method} model...")
    
    with torch.no_grad():
        for H_batch in tqdm(test_loader, desc="Testing"):
            H_batch = H_batch.to(device)
            batch_size = H_batch.shape[0]
            
            for i in range(batch_size):
                H_sample = H_batch[i:i+1]  # [1, num_users, 1, Nt]

                nominated_mask, nominated_mask_torch = build_nomination_mask(
                    H_sample,
                    model,
                    method,
                    input_type,
                    device,
                    nomination_mode=nomination_mode,
                    random_fb_prob=random_fb_prob,
                )

                # User scheduling
                if model.scheduling_method == 'random':
                    final_actions = random_scheduling(nominated_mask_torch, K)
                elif model.scheduling_method == 'exhaustive':
                    final_actions, _ = exhaustive_scheduling(H_sample, K, model.beamforming_method, noise_power)
                elif model.scheduling_method == 'pf':
                    weights = torch.rand_like(nominated_mask_torch) + 0.1
                    final_actions = pf_scheduling(H_sample, nominated_mask_torch, weights, K)
                elif model.scheduling_method == 'sus':
                    alpha = 0.3
                    # import pdb; pdb.set_trace()
                    final_actions = sus_scheduling(nominated_mask_torch, H_sample, K, alpha)
                elif model.scheduling_method == 'greedy':
                    final_actions = topK_scheduling(nominated_mask_torch, H_sample, K)
                else:
                    raise ValueError(f"Invalid scheduling method: {model.scheduling_method}")
                    # final_actions = topK_scheduling(nominated_mask_torch, H_sample, K)
                
                # Beamforming
                if (model.beamforming_method == 'zf'):
                    F_beamforming = zf_beamforming(H_sample, final_actions)
                elif (model.beamforming_method == 'rzf'):
                    F_beamforming = rzf_beamforming(H_sample, final_actions, noise_power)
                else:
                    raise ValueError(f"Invalid beamforming method: {model.beamforming_method}")
                    # F_beamforming = zf_beamforming(H_sample, final_actions)
                
                # Compute sum rate
                sum_rate, avg_rate = sum_rate_calculation(H_sample, final_actions, F_beamforming, noise_power)
                
                # Compute additional metrics
                orthogonality_defect = compute_orthogonality_defect(H_sample, nominated_mask)
                condition_number = compute_condition_number(H_sample, nominated_mask)
                
                all_sum_rates.append(sum_rate.item())
                all_nominated_counts.append(nominated_mask.sum())
                all_scheduled_counts.append(final_actions.sum().item())
                all_orthogonality_defects.extend(orthogonality_defect)
                all_condition_numbers.extend(condition_number)
    
    # Compute statistics
    metrics = {
        'sum_rate_mean': np.mean(all_sum_rates),
        'sum_rate_std': np.std(all_sum_rates),
        'sum_rate_median': np.median(all_sum_rates),
        'nominated_count_mean': np.mean(all_nominated_counts),
        'nominated_count_std': np.std(all_nominated_counts),
        'scheduled_count_mean': np.mean(all_scheduled_counts),
        'scheduled_count_std': np.std(all_scheduled_counts),
        'orthogonality_defect_mean': np.mean(all_orthogonality_defects),
        'orthogonality_defect_std': np.std(all_orthogonality_defects),
        'condition_number_mean': np.mean(all_condition_numbers),
        'condition_number_std': np.std(all_condition_numbers),
        'all_sum_rates': np.array(all_sum_rates),
        'all_nominated_counts': np.array(all_nominated_counts),
        'all_scheduled_counts': np.array(all_scheduled_counts),
        'all_orthogonality_defects': np.array(all_orthogonality_defects),
        'all_condition_numbers': np.array(all_condition_numbers)
    }
    
    return metrics


def save_results(filename, metrics, additional_params):
    """Save evaluation results to a .mat file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    spio.savemat(filename, {**metrics, **additional_params})


def main():
    parser = argparse.ArgumentParser(description='Unified test script for RL communication systems')
    parser.add_argument('--method', type=str, choices=['reinforce', 'directgrad', 'baseline'], 
                       default='reinforce', help='Training method: reinforce, directgrad, or baseline')
    parser.add_argument('--input_type', type=str, choices=['full', 'chg_input'], 
                       default='full', help='Input type: full channel matrix or channel gain only (ignored for baseline)')
    parser.add_argument('--scheduling', type=str, choices=['random', 'greedy', 'exhaustive', 'pf', 'sus'], 
                       default='greedy', help='Scheduling method: random, greedy, exhaustive, pf, or sus')
    parser.add_argument('--beamforming', type=str, choices=['zf', 'rzf'], 
                       default='rzf', help='Beamforming method: zf (Zero Forcing) or rzf (Regularized ZF)')
    parser.add_argument('--channel_mode', type=str, choices=['RF', 'UMi_UPA', 'UMi_ULA', 'Berlin_UPA', 'RMa_UPA'], 
                       default='UMi_UPA', help='Channel mode: RF (Rayleigh Fading), UMi_UPA (3GPP UMi UPA), UMi_ULA (3GPP UMi ULA), Berlin_UPA (3GPP Berlin UMa UPA), or RMa_UPA (3GPP RMa UPA)')
    parser.add_argument('--model_path', type=str, required=False,
                       help='Path to the trained model file (optional for baseline methods)')
    parser.add_argument('--num_test_samples', type=int, default=2000,
                       help='Number of test samples to evaluate')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path for results (default: auto-generated)')
    parser.add_argument('--gpu_id', type=str, default=None,
                       help='GPU ID to use (e.g., "0", "1", "2", "3"). If not specified, uses system default.')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for processing (compatibility argument, not used in CPU version)')
    parser.add_argument('--snr_db', type=float, default=SNR_dB,
                       help='SNR in dB used to compute noise power during evaluation')
    parser.add_argument('--nomination_mode', type=str, choices=['model', 'all_fb', 'random_fb'],
                       default=None, help='Nomination rule for evaluation. Defaults to model for learned methods and all_fb for baselines.')
    parser.add_argument('--random_fb_prob', type=float, default=0.5,
                       help='Nomination probability used when --nomination_mode=random_fb')
    
    args = parser.parse_args()
    
    # Set CUDA device if specified
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"Using GPU: {args.gpu_id}")

    noise_power = 1 / (10 ** (0.1 * args.snr_db))
    if args.nomination_mode is None:
        args.nomination_mode = 'all_fb' if args.method == 'baseline' else 'model'
    
    # Setup
    device = initialize_device()
    
    # Generate output filename if not provided
    if args.output_file is None:
        method_suffix = args.method
        if args.method == 'baseline':
            method_suffix = f"baseline_{args.scheduling}"
        elif args.method in ['reinforce', 'directgrad']:
            method_suffix = f"{args.method}_{args.input_type}"
        
        # Channel mode suffix
        if args.channel_mode == 'RF':
            channel_suffix = 'RF'
        elif args.channel_mode == 'UMi_UPA':
            channel_suffix = 'UMi_UPA'
        elif args.channel_mode == 'UMi_ULA':
            channel_suffix = 'UMi_ULA'
        elif args.channel_mode == 'Berlin_UPA':
            channel_suffix = 'Berlin_UPA'
        elif args.channel_mode == 'RMa_UPA':
            channel_suffix = 'RMa_UPA'
        else:
            channel_suffix = 'UMi_UPA'
        
        # Generate proper filename and save in the correct directory
        if args.method == 'baseline':
            nomination_suffix = "allfb"
            if args.nomination_mode == "random_fb":
                nomination_suffix = f"randomfb_p{str(args.random_fb_prob).replace('.', 'p')}"
            filename = (
                f"baseline_{nomination_suffix}_{args.scheduling}_{args.beamforming}_{channel_suffix}"
                f"_UE{num_users}_M{M}_K{K}_SNR{int(args.snr_db)}_results.mat"
            )
        else:
            filename = (
                f"{method_suffix}_{args.scheduling.upper()}_{args.beamforming.upper()}_{channel_suffix}"
                f"_UE{num_users}_M{M}_K{K}_SNR{int(args.snr_db)}_results.mat"
            )
        args.output_file = os.path.join(save_testresult_path, filename)
    
    print(f"Testing with method: {args.method}, input_type: {args.input_type}")
    print(f"Scheduling: {args.scheduling}, Beamforming: {args.beamforming}")
    print(f"Channel mode: {args.channel_mode}")
    print(f"SNR (dB): {args.snr_db}")
    print(f"Nomination mode: {args.nomination_mode}")
    print(f"Model path: {args.model_path}")
    print(f"Output file: {args.output_file}")
    
    # Load dataset and DataLoader
    dataset = CommunicationDataset(args.channel_mode)
    
    # Calculate test data range for transparency
    if args.channel_mode in ["UMi_UPA", "UMi_ULA"]:
        test_start = num_training_samples + num_val_samples
        test_end = test_start + args.num_test_samples
        print(f"{args.channel_mode} test data range: samples {test_start} to {test_end} (no buffer)")
    else:
        test_start = num_training_samples + num_val_samples
        test_end = test_start + args.num_test_samples
        print(f"RF test data range: samples {test_start} to {test_end}")
    
    test_loader = load_test_data(
        dataset, num_training_samples, num_val_samples, args.num_test_samples, args.channel_mode
    )

    # Load model
    model = load_model(args.model_path, args.method, args.input_type, 
                      args.scheduling, args.beamforming, device)

    # Evaluate model
    metrics = evaluate_model(
        model,
        test_loader,
        args.method,
        device,
        input_type=args.input_type,
        noise_power=noise_power,
        nomination_mode=args.nomination_mode,
        random_fb_prob=args.random_fb_prob,
    )

    # Save results
    additional_params = {
        "Out_epochs": num_epochs,
        "batch_size": args.num_test_samples,
        "method": args.method,
        "input_type": args.input_type,
        "scheduling": args.scheduling,
        "beamforming": args.beamforming,
        "channel_mode": args.channel_mode,
        "snr_db": args.snr_db,
        "noise_power": noise_power,
        "nomination_mode": args.nomination_mode,
        "random_fb_prob": args.random_fb_prob,
    }
    save_results(args.output_file, metrics, additional_params)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Input Type: {args.input_type}")
    print(f"Scheduling: {args.scheduling}")
    print(f"Beamforming: {args.beamforming}")
    print(f"Channel Mode: {args.channel_mode}")
    print(f"Sum Rate - Mean: {metrics['sum_rate_mean']:.4f}, Std: {metrics['sum_rate_std']:.4f}")
    print(f"Nominated Users - Mean: {metrics['nominated_count_mean']:.2f}, Std: {metrics['nominated_count_std']:.2f}")
    print(f"Scheduled Users - Mean: {metrics['scheduled_count_mean']:.2f}, Std: {metrics['scheduled_count_std']:.2f}")
    print(f"Orthogonality Defect - Mean: {metrics['orthogonality_defect_mean']:.4f}, Std: {metrics['orthogonality_defect_std']:.4f}")
    print(f"Condition Number - Mean: {metrics['condition_number_mean']:.4f}, Std: {metrics['condition_number_std']:.4f}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
