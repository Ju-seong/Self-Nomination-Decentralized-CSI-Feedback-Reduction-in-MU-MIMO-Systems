"""
REINFORCE-based self-nomination with full channel matrix input.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from baseline_methods.sched_bf_modules import (
    random_scheduling,
    topK_scheduling,
    zf_beamforming,
    rzf_beamforming,
    sum_rate_calculation,
)


class Modules(nn.Module):
    def __init__(self, scheduling_method="greedy", beamforming_method="rzf"):
        super().__init__()
        self.scheduling_method = scheduling_method
        self.beamforming_method = beamforming_method
        self.deep_sched = nn.Sequential(
            nn.Conv1d(2, 16, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Flatten(),
            nn.Linear(64 * Nt, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.sharpness = 10.0
        self.optimizer = optim.Adam(self.parameters(), lr=my_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.dual_var = torch.tensor(3.0, requires_grad=False)
        self.dual_var_step = torch.tensor(0.001, requires_grad=False)
        self.Aw = Aw

    def forward(self, x):
        return self.deep_sched(x)

    def train_module(self, H_batch, training=True):
        self.train() if training else self.eval()
        device = H_batch.device
        self.dual_var = self.dual_var.to(device)
        self.dual_var_step = self.dual_var_step.to(device)

        H_real = H_batch.real
        H_imag = H_batch.imag
        H_combined = torch.cat((H_real, H_imag), dim=2)
        H_input = H_combined.view(-1, 2, Nt).float()

        outputs = self.forward(H_input) * self.sharpness
        actual_batch_size, actual_num_users = H_batch.shape[0], H_batch.shape[1]
        sigmoid_output = torch.sigmoid(outputs).view(actual_batch_size, actual_num_users)

        with torch.no_grad():
            actions = torch.bernoulli(sigmoid_output)

        log_probs = (
            actions * torch.log(sigmoid_output + 1e-9)
            + (1 - actions) * torch.log(1 - sigmoid_output + 1e-9)
        )
        log_probs_sum = log_probs.sum(dim=1)

        if self.scheduling_method == "random":
            final_actions = random_scheduling(actions, K)
        else:
            final_actions = topK_scheduling(actions, H_batch, K)

        if self.beamforming_method == "zf":
            F_beamforming = zf_beamforming(H_batch, final_actions)
        else:
            F_beamforming = rzf_beamforming(H_batch, final_actions, noise_pwr)

        sum_rate, avg_rate = sum_rate_calculation(H_batch, final_actions, F_beamforming, noise_pwr)
        num_ones_each = actions.sum(dim=1)
        advantage = sum_rate - self.dual_var * (num_ones_each - M)
        advantage_detached = advantage.detach()
        reinforce_loss = -(advantage_detached) * log_probs_sum
        loss = reinforce_loss.mean()

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_ones = num_ones_each.mean()
            self.dual_var += self.dual_var_step * (mean_ones - M)
            self.dual_var = torch.clamp(self.dual_var, min=0)
            self.dual_var = torch.tensor(self.dual_var.item(), requires_grad=False)
            self.dual_var_step *= 0.999

        return loss.item(), avg_rate.item(), actions.sum(dim=1).mean().item()
