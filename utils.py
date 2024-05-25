import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


CLIP = "clip"
L2NORM = "l2norm"
L1NORM = "l1norm"
MAXNORM = "maxnorm"
NONORM = "nonorm"

def normalize_weights(weights, norm_type, dim=0):
    if norm_type == CLIP:
        return weights.clip(-10, 10)
    elif norm_type == L2NORM:
        return weights / (torch.linalg.norm(weights, dim=dim, ord=2) + 1e-30)
    elif norm_type == L1NORM:
        return weights / (torch.linalg.norm(weights, dim=dim, ord=1) + 1e-30)
    elif norm_type == MAXNORM:
        return weights / (torch.abs(weights).amax() + 1e-30)
    elif norm_type == NONORM:
        return weights
    else:
        raise ValueError(f"Invalid norm type: {norm_type}")
    

        
class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]
