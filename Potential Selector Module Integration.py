import torch
import torch.nn as nn

class Selector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, policy_action, feedback, policy_covariance):
        G = policy_covariance @ torch.inverse(policy_covariance + torch.eye(policy_covariance.size(0)))
        corrected_action = policy_action + (G @ feedback)
        return corrected_action
