import torch
from torch.nn.modules.utils import _pair
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision

class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 1,
            initial_lr: float = 0.01,
            device = 'cpu',
            first_layer=False,
            neuron_centric=False,
            learn_t_invert=False
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

        if learn_t_invert:
            self.t_invert = nn.Parameter(torch.tensor(t_invert))
        else:
            self.t_invert = torch.tensor(t_invert)
        
        self.initial_lr = initial_lr
        self.weight_norm_dep = 1
        self.first_layer = first_layer
        self.neuron_centric = neuron_centric

        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        weight = weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)).requires_grad_(False).to(device)

        self.register_buffer('weight', weight)

        if self.neuron_centric:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.Ci = nn.Parameter(torch.ones(in_channels // groups) * 0.01, requires_grad=True)
            self.Cj = nn.Parameter(torch.ones(out_channels) * 0.01, requires_grad=True)
            self.Ck1 = nn.Parameter(torch.ones(kernel_size) * 0.01, requires_grad=True)
            self.Ck2 = nn.Parameter(torch.ones(kernel_size) * 0.01, requires_grad=True)

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        
        if self.training:
            self.weight = self.weight.detach()
            # self.t_invert = self.t_invert.detach()

            weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

            # perform conv, obtain weighted input u \in [B, OC, OH, OW]
            
            # ===== find post-synaptic activations y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # Post-synaptic activation, for plastic update, is weighted input passed through a softmax.
            # Non-winning neurons (those not with the highest activation) receive the negated post-synaptic activation.
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)

            if True:
                # Compute the winner neuron for each batch element and pixel
                flat_softwta_activs = - flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
                win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
                competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
                # Turn winner neurons' activations back to hebbian
                flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]

            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # ===== compute plastic update Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # Use Convolutions to apply the plastic update. Sweep over inputs with postynaptic activations.
            # Each weighting of an input pixel & an activation pixel updates the kernel element that connected them in
            # the forward pass.
            yx = F.conv2d(
                x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
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
                eta = self.Cj[:, None, None, None] * self.Ci[None, :, None, None] * self.Ck1[None, None, :, None] * self.Ck2[None, None, None, :]
            else:
                eta = torch.abs(torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                eta = (eta ** 0.5)[:, None, None, None] * self.initial_lr   

            self.weight = self.weight + delta_weight * eta 
            self.weight = self.weight.clip(-10, 10)
            # if self.t_invert < 100 and self.t_invert > 0.01:
                # self.t_invert *= 1.01
                # self.t_invert *= 0.999

        return F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)
        
    
class SoftHebbLinear(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            device = 'cpu',
            activation = None,
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

        self.act = activation
        
        weight = torch.empty((out_channels, in_channels)).requires_grad_(False).to(device)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        self.register_buffer('weight', weight)

        self.Ci = nn.Parameter(torch.ones(1, in_channels) * 0.1, requires_grad=True)
        self.Cj = nn.Parameter(torch.ones(out_channels, 1) * 0.01, requires_grad=True)

    def forward(self, x, target = None):
        # x = x / x.norm(dim=1, keepdim=True)
        out = F.linear(x, self.weight)
        if self.training and target is not None: 
            # clean gradients
            self.weight = self.weight.detach()
                        
            xt = target.T @ x
            out = (out).softmax(dim=1)
            xy = out.T @ x

            delta_weight = (xt - xy).detach()
            
            self.weight = self.weight + delta_weight.detach() * self.Ci * self.Cj
            # self.weight = self.wanda_prune(self.weight, x, 0.3)
            self.weight = self.weight.clip(-10, 10)
            return F.linear(x, self.weight)

        return out


class DeepSoftHebb(nn.Module):
    def __init__(self, device = 'cpu', in_channels=1, dropout=0.0, input_size=32, neuron_centric=False, unsupervised_first=False, learn_t_invert=False):
        super(DeepSoftHebb, self).__init__()
        # block 1
        self.bn1 = nn.BatchNorm2d(in_channels, affine=False).requires_grad_(False)
        self.conv1 = SoftHebbConv2d(
            in_channels=in_channels, 
            out_channels=96, 
            kernel_size=5, 
            padding=2, 
            t_invert=1., 
            device=device, 
            initial_lr=0.08, 
            first_layer=True,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert
        )
        self.activ1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False).requires_grad_(False)
        self.conv2 = SoftHebbConv2d(
            in_channels=96, 
            out_channels=384,
            kernel_size=3, 
            padding=1, 
            t_invert=1., 
            device=device, 
            initial_lr=0.005,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert
        )
        self.activ2 = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 3

        self.bn3 = nn.BatchNorm2d(384, affine=False).requires_grad_(False)
        self.conv3 = SoftHebbConv2d(
            in_channels=384, 
            out_channels=1536,
            kernel_size=3, 
            padding=1, 
            t_invert=1., 
            device=device,
            initial_lr=0.01,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert
        )

        self.activ3 = nn.ReLU() #Triangle(power=1.) #
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # block 4
        self.flatten = nn.Flatten()

        if neuron_centric:
            self.classifier = SoftHebbLinear((input_size // 2) * 1536, 10, device=device)
        else:
            self.classifier = nn.Linear((input_size // 2) * 1536, 10)
            self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, (input_size // 2) * 1536)

        self.neuron_centric = neuron_centric
        self.unsupervised_first = unsupervised_first
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, target=None):
        if target is not None:
            target = F.one_hot(target, 10).float().to(x.device) # - 0.1
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))
        # block 4
        if self.neuron_centric:
            return self.classifier(self.dropout(self.flatten(out)), target)
        else:
            return self.classifier(self.dropout(self.flatten(out)))
        
    def unsupervised_eval(self):
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()
    

class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power
    

    
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
