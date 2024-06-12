
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.nn.grad
from utils import normalize_weights, CLIP, balanced_credit_fn, L2NORM, L1NORM, MAXNORM, SOFTMAX, RELU, TANH, SOFTMAX

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
            label_smoothing=None,
            activation=RELU
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
        self.label_smoothing = label_smoothing
        self.activation = activation

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
        if self.training:
            self.weight = self.weight.detach()
            eta = self.Ci * self.Cj * self.lr

            #if target is not None and target.shape[-1] != 10:

            if target is not None and self.last_layer: 
                target = torch.functional.F.one_hot(target, 10).float().to(self.x.device)
                dw = self.target_update_rule(target)
            else:
                dw = self.update_rule(target=target)
            if self.use_momentum:
                if self.momentum is not None:
                    self.momentum = self.momentum * 0.9 + dw.detach() * 0.1
                else:
                    self.momentum = dw.detach()
                self.weight = self.weight + self.momentum.detach() * eta
            else:
                self.weight = self.weight + dw.detach() * eta
            
            #self.weight = self.weight + dw.detach() * eta
            self.weight = normalize_weights(self.weight, L2NORM, dim=0)

            self.x = None
            self.out = None

    def target_update_rule(self, target):
        out = self.out.softmax(dim=1)
        return (target - out).T @ self.x

    def update_rule(self, target):
        if self.activation == RELU:
            out = F.relu(self.out)
        elif self.activation == TANH:
            out = torch.tanh(self.out)
        elif self.activation == SOFTMAX:
            out = self.out

        norms = torch.norm(out, dim=1, keepdim=True, p=2) + 1e-8
        out_norm = out / norms

        # Cosine similarity matrix
        similarity = out_norm @ out_norm.T # [batch, output_channels, output_channels]

        # Create masks based on class labels
        positive_mask = (target.unsqueeze(1) == target.unsqueeze(0)).float()
        negative_mask = 1.0 - positive_mask
        # set the diagonal to 0, negative_mask is already 0 there
        positive_mask.fill_diagonal_(0)

        # The loss aims to minimize the similarity between samples of the same class
        # and maximize the similarity between samples of different classes
        # loss = (positive_mask * (1 - similarity) + negative_mask * similarity.abs()).mean()

        # Add label smoothing to positive and negative masks
        if self.label_smoothing:
            positive_mask = (1 - self.label_smoothing) * positive_mask + self.label_smoothing / (self.out.shape[0] - 1)
            negative_mask = (1 - self.label_smoothing) * negative_mask + self.label_smoothing / (self.out.shape[0] - 1)
            positive_mask.fill_diagonal_(0)
            negative_mask.fill_diagonal_(0)


        # Compute the gradient of the loss w.r.t. similarity
        grad_similarity = negative_mask * similarity.sign() - positive_mask
        grad_similarity /= similarity.numel()

        # Compute gradient of similarity w.r.t. normalized output (out_norm)
        grad_out = grad_similarity @ out_norm + (grad_similarity.T @ out_norm)

        # Backpropagate through normalization
        grad_out = (grad_out * norms - out_norm * torch.sum(out * grad_out, dim=1, keepdim=True)) / norms**2
        # grad_out = grad_out_norm / norms ** 2
    
        # Backpropagate through tanh
        if self.activation == RELU:
            grad_out *= (out > 0).float().view_as(grad_out)
        elif self.activation == TANH:
            grad_out *= (1 - torch.tanh(out)**2).view_as(grad_out)
        elif self.activation == SOFTMAX:
            grad_out *= out

        # Compute the gradient of the loss w.r.t. the input
        grad_weight = grad_out.T @ self.x
        return - grad_weight