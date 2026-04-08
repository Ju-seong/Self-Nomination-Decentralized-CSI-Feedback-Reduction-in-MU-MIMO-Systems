"""
Direct gradient self-nomination with channel gain input (straight-through estimator).
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
    zf_beamforming_alt,
    rzf_beamforming,
    sum_rate_calculation,
)


class Modules(nn.Module):
    def __init__(self, scheduling_method="greedy", beamforming_method="zf"):
        super().__init__()
        self.scheduling_method = scheduling_method
        self.beamforming_method = beamforming_method
        self.deep_sched = nn.Sequential(
            nn.Linear(1, Nt),
            nn.ELU(),
            nn.BatchNorm1d(Nt),
            nn.Linear(Nt, Nt),
            nn.ELU(),
            nn.BatchNorm1d(Nt),
            nn.Linear(Nt, 1),
            nn.Tanh(),
        )
        self.sharpness = 10.0
        self.optimizer = optim.Adam(self.parameters(), lr=my_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.dual_var = torch.tensor(3.0, requires_grad=False)
        self.dual_var_step = torch.tensor(0.0001, requires_grad=False)
        self.Aw = Aw

    def forward(self, x):
        return self.deep_sched(x)

    def train_module(self, H_batch, training=True):
        self.train() if training else self.eval()
        device = H_batch.device
        self.dual_var = self.dual_var.to(device)
        self.dual_var_step = self.dual_var_step.to(device)

        chg = torch.norm(H_batch.squeeze(2), dim=2)
        chg_input = chg.view(-1, 1).float()
        outputs = self.forward(chg_input) * self.sharpness
        actual_batch_size, actual_num_users = H_batch.shape[0], H_batch.shape[1]
        sigmoid_output = torch.sigmoid(outputs).view(actual_batch_size, actual_num_users)
        binary_output = ((sigmoid_output > 0.5).float() - sigmoid_output).detach() + sigmoid_output
        actions = binary_output.view(actual_batch_size, actual_num_users)

        if self.scheduling_method == "random":
            actions_mask = random_scheduling(actions, K).detach()
        else:
            actions_mask = topK_scheduling(actions, H_batch, K).detach()
        final_actions = actions * actions_mask
        num_ones_each = actions.sum(dim=1)

        if self.beamforming_method == "zf":
            F_beamforming = zf_beamforming_alt(H_batch, final_actions)
        else:
            F_beamforming = rzf_beamforming(H_batch, final_actions, noise_pwr)

        sum_rate, avg_rate = sum_rate_calculation(H_batch, final_actions, F_beamforming, noise_pwr)
        loss = -avg_rate + self.dual_var * (num_ones_each.mean() - M)

        if training:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.dual_var += self.dual_var_step * (num_ones_each.mean() - M)
            self.dual_var = torch.clamp(self.dual_var, min=0)
            self.dual_var = torch.tensor(self.dual_var.item(), requires_grad=False)
            self.dual_var_step *= 0.995

        return loss.item(), avg_rate.item(), num_ones_each.mean().item()
