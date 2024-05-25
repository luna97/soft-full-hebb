import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _pair
from utils import normalize_weights, CLIP

class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 1.,
            initial_lr: float = 0.01,
            device = 'cpu',
            first_layer=False,
            neuron_centric=False,
            learn_t_invert=False,
            norm_type=CLIP,
            two_steps=False
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        super(SoftHebbConv2d, self).__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        self.norm_type = norm_type

        if learn_t_invert:
            self.t_invert = nn.Parameter(torch.tensor(t_invert, dtype=torch.float), requires_grad=True)
        else:
            self.t_invert = torch.tensor(t_invert)
        
        self.initial_lr = initial_lr
        self.first_layer = first_layer
        self.neuron_centric = neuron_centric
        self.two_steps = two_steps

        self.momentum = None

        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        weight = weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)).requires_grad_(False).to(device)

        self.register_buffer('weight', weight)

        if self.neuron_centric:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.Ci = nn.Parameter(torch.ones(in_channels // groups) * .1, requires_grad=True)
            self.Cj = nn.Parameter(torch.ones(out_channels) * .1, requires_grad=True)
            self.Ck1 = nn.Parameter(torch.ones(kernel_size) * .1, requires_grad=True)
            self.Ck2 = nn.Parameter(torch.ones(kernel_size) * .1, requires_grad=True)

    def forward(self, x):
        # x = x / (x.norm(dim=1, keepdim=True) + 1e-30)
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        tortn = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)
        self.out = tortn.clone().detach()
        self.x = x.clone().detach()
        if self.two_steps:
            self.step()
            return F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)
        return tortn
    
    def step(self):
        if self.training:
            self.weight = self.weight.detach()
            # self.t_invert = self.t_invert.detach()
            # weighted_input = F.conv_transpose2d(self.x, self.weight, None, self.stride, 0, self.dilation, self.groups)
            
            weighted_input = self.out

            # perform conv, obtain weighted input u \in [B, OC, OH, OW]
            
            # ===== find post-synaptic activations y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # Post-synaptic activation, for plastic update, is weighted input passed through a softmax.
            # Non-winning neurons (those not with the highest activation) receive the negated post-synaptic activation.
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)


            if True:
                win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
                # soft = torch.softmax(flat_softwta_activs.T * 100, dim=1)
                # print(math.log(self.out_channels))
                # topk_values, topk_indices = torch.topk(flat_softwta_activs, k=3, dim=0)
                # seconds_winners = topk_indices[1]  # winning kernel for each pixel in each input 
                # third_winners = topk_indices[2]  # winning kernel for each pixel in each input

                # Compute the winner neuron for each batch element and pixel
                flat_softwta_activs = - flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian

                competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
                # Turn winner neurons' activations back to hebbian
                flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]
                # flat_softwta_activs[seconds_winners, competing_idx] = -flat_softwta_activs[seconds_winners, competing_idx]
                # flat_softwta_activs[third_winners, competing_idx] = -torch.mean(flat_softwta_activs, dim=0)



            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # ===== compute plastic update Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # Use Convolutions to apply the plastic update. Sweep over inputs with postynaptic activations.
            # Each weighting of an input pixel & an activation pixel updates the kernel element that connected them in
            # the forward pass.
            yx = F.conv2d(
                self.x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
                softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
                padding=0,
                stride=self.dilation,
                dilation=self.stride,
                groups=1
            ).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)

            # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight.detach()
            delta_weight = delta_weight / (torch.abs(delta_weight).amax() + 1e-30)  # Scale [min/max , 1]

            if self.neuron_centric:
                # eta = torch.abs(torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                # eta = (eta ** 0.5)[:, None, None, None]
                eta = self.Cj[:, None, None, None] * self.Ci[None, :, None, None] * self.Ck1[None, None, :, None] * self.Ck2[None, None, None, :]
                # eta = eta * cicj
            else:
                eta = torch.abs(torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                eta = (eta ** 0.5)[:, None, None, None] * self.initial_lr   

            if self.momentum is None:
                self.momentum = delta_weight
            else:
                self.momentum = 0.9 * self.momentum.detach() + 0.1 * delta_weight

            self.weight = self.weight.detach() + self.momentum * eta 
            self.weight = normalize_weights(self.weight, self.norm_type, dim=0)
            self.x = None
            self.out = None