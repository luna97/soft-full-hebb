import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.grad
from torch.nn.modules.utils import _pair
from utils import normalize_weights, CLIP, balanced_credit_fn, L2NORM, L1NORM, MAXNORM

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
            t_invert: float = 1.,
            initial_lr: float = 0.01,
            device = 'cpu',
            first_layer=False,
            neuron_centric=False,
            learn_t_invert=False,
            norm_type=CLIP,
            two_steps=False,
            inp_size=None
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

        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        weight = weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)).requires_grad_(False).to(device)

        self.register_buffer('weight', weight)

        self.inp_size = inp_size ** 2 * in_channels if inp_size is not None else None

        self.momentum = None

        if self.neuron_centric:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.Ci = nn.Parameter(torch.ones(in_channels // groups), requires_grad=True)
            self.Cj = nn.Parameter(torch.ones(out_channels), requires_grad=True)
            self.Ck1 = nn.Parameter(torch.ones(kernel_size), requires_grad=True)
            self.Ck2 = nn.Parameter(torch.ones(kernel_size), requires_grad=True)

    def forward(self, x, target = None):
        x = x / (x.norm(dim=1, keepdim=True) + 1e-30)
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        tortn = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        # keeping only the sum of the batches for memory efficiency
        self.out = tortn.clone().detach()# .sum(dim=0).unsqueeze(0)
        self.x = x.clone().detach()#.sum(dim=0).unsqueeze(0)
        if self.two_steps:
            self.step(target=target)
            return F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)
        return tortn
    
    def step(self, credit=None, target=None):
        if self.training:  
            self.weight = self.weight.detach()
            
            if credit is not None:
                dw = self.credit_update_rule(credit)
                self.update_credit(credit)
            elif target is not None:
                # dw = self.regularize_orthogonality() * 0.01
                # dw = self.update_rule_channel(target=target)
                dw = self.update_rule_sample(target=target)
                #dw = (dw1 + dw2) / 2
  
            if self.neuron_centric:
                # eta = torch.abs(torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                # eta = (eta ** 0.5)[:, None, None, None]
                eta = self.Cj[:, None, None, None] * self.Ci[None, :, None, None] * self.Ck1[None, None, :, None] * self.Ck2[None, None, None, :]
                # eta = eta * cicj
            else:
                eta = torch.abs(torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                eta = (eta ** 0.5)[:, None, None, None] * self.initial_lr  


            # if self.momentum is None:
            #    self.momentum = dw
            #else:
            #    self.momentum = 0.9 * self.momentum + 0.1 * dw 

            # print(self.momentum.norm())
            # reg = self.regularize_orthogonality()
            # self.weight = self.weight + (self.momentum.detach())* eta * 0.001 
            self.weight = self.weight + dw.detach() * eta * 0.0001
            # print(eta.norm())
            self.weight = normalize_weights(self.weight, self.norm_type, dim=[2, 3])
            self.x = None
            self.out = None

    def soft_hebb_rule(self, antihebb=False):
        batch_size, out_channels, height_out, width_out = self.out.shape
        # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
        flat_weighted_inputs = self.out.transpose(0, 1).reshape(out_channels, -1)
        # Compute the winner neuron for each batch element and pixel
        flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)

        if antihebb:
            flat_softwta_activs = - flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
            competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
            # Turn winner neurons' activations back to hebbian
            flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]

        softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
        yx = torch.nn.grad.conv2d_weight(
            self.x, #.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
            self.weight.shape,
            softwta_activs, #.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
            padding=0,
            stride=self.stride,
            dilation=self.dilation,
            groups=1
        )
        #.transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)
        # yx = F.conv2d(
        #    self.x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
        #    softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
        #    padding=0,
        #    stride=self.dilation,
        #    dilation=self.stride,
        #    groups=1
        #).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)

        # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
        yu = torch.sum(torch.mul(softwta_activs, self.out), dim=(0, 2, 3))
        delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight.detach()
        # delta_weight = delta_weight / (torch.abs(delta_weight).amax() + 1e-30)  # Scale [min/max , 1]
        return delta_weight
    
    def credit_update_rule(self, credit):
        return torch.zeros_like(self.weight)
        batch_size, out_channels, height_out, width_out = self.out.shape
        batch_size, in_channels, height_in, width_in = self.x.shape
        # credit is flattened and corresponds to [out_channels * out_img_size ** 2]
        # credit = credit.view(self.out_channels, height_out, width_out)# .flip((1, 2))

        balance = balanced_credit_fn(self.out.view(batch_size, -1)) # [batch, out_channels]
        c_bal = credit[None, :] * balance # [batch, out_channels, height_out, width_out]

        c_bal = c_bal.view(batch_size, out_channels, height_out, width_out)

        return - torch.nn.grad.conv2d_weight(
            self.x,
            self.weight.shape,
            c_bal,
            stride=self.stride,
            padding=self.F_padding[0],
            dilation=self.dilation,
            groups=1
        ) / batch_size

    def update_credit(self, credit):
        with torch.no_grad():
            balance = balanced_credit_fn(self.out) # [batch, out_channels * height_out * width_out]

            batch_size, out_channels, height_out, width_out = self.out.shape
            in_channels, height_in, width_in = self.x.shape[1:]
            credit = credit.view(1, out_channels, height_out, width_out)
            # credit is at output level
            c_bal = credit * balance # [batch, out_channels, height_out, width_out]

            # this is equal to the backward pass of a linear layer for the inputs
            J = torch.nn.grad.conv2d_input(
                self.x.shape,
                self.weight, #.flip((2, 3)),
                c_bal,
                self.stride, 
                self.F_padding[0],
                self.dilation,
                self.groups
            )

            self.credit = J.view(batch_size, -1).mean(dim=0) 
            # self.credit = normalize_weights(self.credit, L1NORM, dim=0)

    def update_rule_channel(self, target):
        # Reshape the output to [batch_size, output_channels * height * width]
        batch_size, out_channels, height, width = self.out.shape
        out = self.out.view(batch_size, out_channels, -1) # [batch, output_channels, height * width]

        # Cosine similarity matrix
        similarity = torch.bmm(out, out.transpose(2, 1))  # [batch, output_channels, output_channels]
        out_norm = torch.linalg.norm(out, dim=2, ord=2).to(self.out.device)  # [batch, output_channels]
        similarity = similarity / (out_norm[:, :, None] * out_norm[:, None, :]) # [batch, output_channels, output_channels]

        similarity = similarity - torch.diag_embed(torch.diagonal(similarity, dim1=-2, dim2=-1)) # [batch, output_channels, output_channels]

        # Manually calculate the gradient
        grad_similarity = 2 * similarity.abs()  # [batch, batch]

        # Compute the gradient of the loss w.r.t. out
        out_norm_matrix = (out_norm[:, :, None] * out_norm[:, None, :]) # [batch, output_channels, output_channels]

        grad_similarity_normalized = grad_similarity / out_norm_matrix # [batch, output_channels, output_channels]
        grad_out = grad_similarity_normalized @ out


        # Reshape grad_out back to the shape of the convolutional layer's output
        grad_out = grad_out.view(batch_size, out_channels, height, width)


        # Compute the gradient with respect to the convolutional weights
        conv_weight_grad = torch.nn.grad.conv2d_weight(
            self.x, self.weight.shape, grad_out, self.stride, self.F_padding[0], self.dilation, self.groups
        ) / batch_size

        return - conv_weight_grad 
    

    def update_rule_sample(self, target):
        # Reshape the output to [batch_size, output_channels * height * width]
        batch_size, out_channels, height, width = self.out.shape
        out = self.out.view(batch_size, -1) # [batch_size, height * width * output_channels]

        # Cosine similarity matrix
        similarity = out @ out.T # [batch, output_channels, output_channels]
        out_norm = torch.linalg.norm(out, dim=1, ord=2).to(self.out.device)  # [batch_size]
        similarity = similarity / (out_norm[ :, None] * out_norm[None, :]) # [batch_size, batch_size]

        similarity = similarity - torch.diag_embed(torch.diagonal(similarity, dim1=0, dim2=1)) # [batch_size, batch_size]
        eye_mask = torch.eye(out_channels)
        mask = torch.ones((batch_size, out_channels, out_channels)) - eye_mask[None, ...]
        mask = mask.to(self.out.device)

        positive_mask = (target.unsqueeze(1) == target.unsqueeze(0)).float()
        negative_mask = 1.0 - positive_mask
        eye_mask = torch.eye(batch_size).to(positive_mask.device)
        positive_mask -= eye_mask


        # Manually calculate the gradient
        # grad_similarity = 2 * (negative_mask - positive_mask) * similarity.abs() #
        grad_similarity = 2 * (negative_mask - positive_mask) * similarity.abs() #


        # Compute the gradient of the loss w.r.t. out
        out_norm_matrix = (out_norm[:, None] * out_norm[None, :]) # [batch_size, batch_size]

        grad_similarity_normalized = grad_similarity / out_norm_matrix # [batch_size, batch_size]
        grad_out = grad_similarity_normalized @ out


        # Reshape grad_out back to the shape of the convolutional layer's output
        grad_out = grad_out.view(batch_size, out_channels, height, width)


        # Compute the gradient with respect to the convolutional weights
        conv_weight_grad = torch.nn.grad.conv2d_weight(
            self.x, self.weight.shape, grad_out, self.stride, self.F_padding[0], self.dilation, self.groups
        ) / batch_size

        return - conv_weight_grad 
    
    def regularize_orthogonality(self):
        weight = self.weight.view(self.weight.shape[0], -1)
        weight_norm = torch.linalg.norm(weight, dim=1, ord=2)
        weight = weight / weight_norm[:, None]
        similarity = weight @ weight.T
        similarity = similarity - torch.diag(torch.diag(similarity))

        dw_ortho = 2 * similarity.abs() @ weight
        dw_ortho = dw_ortho.view(self.weight.shape)
        return - dw_ortho

        