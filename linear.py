
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.nn.grad
from utils import normalize_weights, CLIP, balanced_credit_fn, L2NORM, L1NORM, MAXNORM, SOFTMAX

class SoftHebbLinear(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            device = 'cpu',
            norm_type=CLIP,
            two_steps=False,
            last_layer=False,
            initial_lr=0.001,
            use_momentum=False,
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

        self.Ci = nn.Parameter(torch.ones(1, in_channels), requires_grad=True)
        self.Cj = nn.Parameter(torch.ones(out_channels, 1), requires_grad=True)
        self.two_steps = two_steps
        self.credit = torch.ones(in_channels).requires_grad_(False).to(device)
        self.last_layer = last_layer
        self.lr = initial_lr
        self.device = device
        self.use_momentum = use_momentum

    def forward(self, x, target = None, credit = None):
        # x = x / (x.norm(dim=1, keepdim=True) + 1e-30)
        tortn = F.linear(x, self.weight)
        self.x = x.clone().detach()
        self.out = tortn.clone().detach()

        # if credit is not None:
        #    self.update_credit(credit)

        if self.two_steps:
            self.step(target, credit)
            return F.linear(x, self.weight)

        return tortn
    
    def step(self, target=None, credit=None):
        if self.training:
            self.weight = self.weight.detach()
            eta = self.Ci * self.Cj * self.lr

            #if target is not None and target.shape[-1] != 10:
                

            if target is not None and self.last_layer: 
                target = torch.functional.F.one_hot(target, 10).float().to(self.x.device)
                dw = self.target_update_rule(target)
                # self.update_credit(target)
            else:
                dw = self.update_rule(target=target)
                # dw = self.credit_update_rule(credit)
                # self.update_credit(credit)
                # dw = self.update_rule()
            if self.use_momentum:
                if self.momentum is not None:
                    self.momentum = self.momentum * 0.9 + dw.detach() * 0.1
                else:
                    self.momentum = dw.detach()
                self.weight = self.weight + self.momentum.detach() * eta
            else:
                self.weight = self.weight + dw.detach() * eta
            
            #self.weight = self.weight + dw.detach() * eta
            self.weight = normalize_weights(self.weight, self.norm_type, dim=0)

            self.x = None
            self.out = None

    def target_update_rule(self, target):
        out = self.out.softmax(dim=1)
        return (target - out).T @ self.x # / self.out.shape[0] 

    def update_rule(self, target):
        self.out = F.tanh(self.out)
        # self.out = 1 - torch.tanh(self.out) ** 2

        similarity = (self.out) @ (self.out).T # [batch, batch]
        out_norm = torch.linalg.norm(self.out, dim=1, ord=2) # [batch]
        similarity = similarity / (out_norm[:, None] * out_norm[None, :]) # [batch, batch]

        diag = torch.diag(torch.diag(similarity)) # [batch, batch]
        similarity = similarity - diag # [batch, batch]
        batch_size = similarity.shape[0]

        # Create masks based on class labels
        positive_mask = (target.unsqueeze(1) == target.unsqueeze(0)).float()
        negative_mask = 1.0 - positive_mask
        eye_mask = torch.eye(batch_size).to(positive_mask.device)
        # mask = torch.ones_like(positive_mask) - eye_mask

        # Ensure no self-similarity is considered
        positive_mask -= eye_mask
        out_norm_matrix = out_norm[:, None] * out_norm[None, :]

        grad_similarity = 2 * (negative_mask - positive_mask) * similarity.abs() #
        grad_similarity = grad_similarity / (out_norm_matrix + 1e-8)
        grad_out = grad_similarity @ self.out
        grad_weight = grad_out.T @ self.x
        #print(grad_weight)
        return - grad_weight / batch_size
    
    def credit_update_rule(self, credit):
        balance = balanced_credit_fn(self.out) # [batch, out_channels]
        c_bal = credit[None, :] * balance # [batch, out_channels]

        # this is equal to the backward pass of a linear layer for the weights
        #return - c_bal.T @ self.x / self.out.shape[0]
        return torch.zeros_like(self.weight)

    

    def update_credit(self, credit):
        with torch.no_grad():
            if self.weight.shape[0] == 10:
                c_bal = credit - self.out # F.softmax(self.out, dim=1)
            else:
                balance = balanced_credit_fn(self.out) # [batch, out_channels]
                c_bal = credit[None, :] * balance # [batch, out_channels]

            # this is equal to the backward pass of a linear layer for the inputs
            J = c_bal @ self.weight # [batch, in_channels]
            
            self.credit = J.mean(dim=0)
            self.credit = normalize_weights(self.credit, L1NORM, dim=0)
            