
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import normalize_weights, CLIP

  
class SoftHebbLinear(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            device = 'cpu',
            norm_type=CLIP,
            two_steps=False
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        super(SoftHebbLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        
        weight = torch.empty((out_channels, in_channels)).requires_grad_(False).to(device)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        self.register_buffer('weight', weight)
        self.momentum = None

        self.Ci = nn.Parameter(torch.ones(1, in_channels) * 0.01, requires_grad=True)
        self.Cj = nn.Parameter(torch.ones(out_channels, 1) * 0.01, requires_grad=True)
        self.two_steps = two_steps

    def forward(self, x, target = None):
        # x = x / (x.norm(dim=1, keepdim=True) + 1e-30)
        tortn = F.linear(x, self.weight)
        self.x = x.clone().detach()
        self.out = tortn.clone().detach()

        if self.two_steps:
            self.step(target)
            return F.linear(x, self.weight)

        return tortn
    
    def step(self, target=None):
        if self.training and target is not None: 
            # clean gradients
            self.weight = self.weight.detach()
            out = (self.out).softmax(dim=1)

            # out[out.max(dim=1).values < 0.2] = 0.
                        
            if target.shape[-1] != 10:
                target = torch.functional.F.one_hot(target, 10).float().to(self.x.device)

            delta_weight = (target - out).T @ self.x # / self.x .shape[0]
            eta = self.Ci * self.Cj
            # delta_weight = delta_weight * eta

            if self.momentum is None:
                self.momentum = delta_weight
            else:
                self.momentum = 0.9 * self.momentum.detach() + 0.1 * delta_weight

            self.weight = self.weight + self.momentum * eta # - 0.001 * self.weight ** 2
            self.weight = normalize_weights(self.weight, self.norm_type, dim=1)
            # self.weight = self.weight.clip(-10, 10)
            # return F.linear(x, self.weight)
            self.x = None
            self.out = None

