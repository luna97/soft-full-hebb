import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.grad
from torch.nn.modules.utils import _pair
from utils import normalize_weights, CLIP, balanced_credit_fn, L2NORM, L1NORM, MAXNORM

# conv rules
SOFTHEBB = 'softhebb'
ANTIHEBB = 'antihebb'
CHANNEL = 'channel'
SAMPLE = 'sample'
CHSAMPLE = 'chsample'

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
            inp_size=None,
            use_momentum=False,
            conv_rule=SAMPLE,
            regularize_orth=False,
            label_smoothing=None
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
        self.regularize_orth = regularize_orth

        if learn_t_invert:
            self.t_invert = nn.Parameter(torch.tensor(t_invert, dtype=torch.float), requires_grad=True)
        else:
            self.t_invert = torch.tensor(t_invert)
        
        self.initial_lr = initial_lr
        self.first_layer = first_layer
        self.neuron_centric = neuron_centric
        self.two_steps = two_steps
        self.use_momentum = use_momentum
        self.conv_rule = conv_rule
        self.label_smoothing = label_smoothing

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
        # x = x / (x.norm(dim=1, keepdim=True) + 1e-30)
        if self.conv_rule == SOFTHEBB or self.conv_rule == ANTIHEBB:
            x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
            self.x = x.clone().detach()
        else:
            self.x = x.clone().detach()
            x = F.pad(x, self.F_padding, self.padding_mode)  # pad input 
        tortn = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        # keeping only the sum of the batches for memory efficiency
        self.out = tortn.clone().detach()# .sum(dim=0).unsqueeze(0)
        # self.x = x.clone().detach()#.sum(dim=0).unsqueeze(0)
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
                if self.conv_rule == SOFTHEBB:
                    dw = self.soft_hebb_rule()
                elif self.conv_rule == ANTIHEBB:
                    dw = self.soft_hebb_rule(antihebb=True)
                elif self.conv_rule == CHANNEL:
                    dw = self.update_rule_channel(target)
                elif self.conv_rule == SAMPLE:
                    dw = self.update_rule_sample(target)
                elif self.conv_rule == CHSAMPLE:
                    dw1 = self.update_rule_channel(target)
                    dw2 = self.update_rule_sample(target)
                    dw = dw1 + dw2
                else:
                    raise ValueError(f"Invalid conv rule: {self.conv_rule}")
  
            if self.neuron_centric:
                eta = self.Cj[:, None, None, None] * self.Ci[None, :, None, None] * self.Ck1[None, None, :, None] * self.Ck2[None, None, None, :] * self.initial_lr
            else:
                eta = torch.abs(torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                eta = (eta ** 0.5)[:, None, None, None] * self.initial_lr  

            if self.use_momentum:
                if self.momentum is not None:
                    self.momentum = self.momentum * 0.9 + dw.detach() * 0.1
                else:
                    self.momentum = dw.detach()
                if self.regularize_orth:
                    self.weight = self.weight + (self.momentum.detach() + self.regularize_orthogonality()) * eta
                else:
                    self.weight = self.weight + self.momentum.detach() * eta
            else:
                if self.regularize_orth:
                    self.weight = self.weight + (dw.detach() + self.regularize_orthogonality()) * eta 
                else:
                    self.weight = self.weight + dw.detach() * eta

            to_norm = self.weight.view(self.weight.shape[0], -1)
            self.weight = normalize_weights(to_norm, self.norm_type, dim=0).view(self.weight.shape)
            self.x = None
            self.out = None

    def soft_hebb_rule(self, antihebb=False):
        batch_size, out_channels, height_out, width_out = self.out.shape
        # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
        flat_weighted_inputs = self.out.transpose(0, 1).reshape(out_channels, -1)
        # Compute the winner neuron for each batch element and pixel
        flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)

        if antihebb:
            flat_softwta_activs = -flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
            competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
            # Turn winner neurons' activations back to hebbian
            flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]

        softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
        yx = F.conv2d(
            self.x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
            softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
            padding=0,
            stride=self.dilation,
            dilation=self.stride,
            groups=1
        ).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)

        # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
        yu = torch.sum(torch.mul(softwta_activs, self.out), dim=(0, 2, 3))
        delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight.detach()
        return delta_weight

    def update_rule_channel(self, target):
        # Reshape the output to [batch_size, output_channels * height * width]
        batch_size, out_channels, height, width = self.out.shape
        out = self.out.reshape(batch_size, out_channels, -1)

        # out = self.out.view(batch_size, out_channels, -1) # [batch, output_channels, height * width]
        out = F.tanh(out)

        norms = torch.norm(out, dim=2, keepdim=True, p=2) + 1e-8
        out_norm = out / norms

        # Cosine similarity matrix
        similarity = torch.bmm(out_norm, out_norm.transpose(2, 1)) # [batch, output_channels, output_channels]

        # Create masks based on class labels
        # positive_mask = (target.unsqueeze(1) == target.unsqueeze(0)).float() # [batch, batch]
        # negative_mask = 1.0 - positive_mask # [batch, batch]
        # set the diagonal to 0, negative_mask is already 0 there
        # positive_mask.fill_diagonal_(0) # [batch, batch]

        channel_mask = torch.ones(out_channels, out_channels).fill_diagonal_(0).unsqueeze(0).to(out.device)

        # The loss aims to minimize the similarity between samples of the same class
        # and maximize the similarity between samples of different classes
        # loss = (positive_mask * (1 - similarity) + negative_mask * similarity.abs()).sum()

        # Add label smoothing to positive and negative masks
        if self.label_smoothing is not None:
            positive_mask = (1 - self.label_smoothing) * positive_mask + self.label_smoothing / (self.out.shape[0] - 1)
            negative_mask = (1 - self.label_smoothing) * negative_mask + self.label_smoothing / (self.out.shape[0] - 1)
            positive_mask.fill_diagonal_(0)
            negative_mask.fill_diagonal_(0)

        # Compute the gradient of the loss w.r.t. similarity
        grad_similarity = channel_mask * similarity.abs()

        # Compute gradient of similarity w.r.t. normalized output (out_norm)
        grad_out_norm = grad_similarity @ out_norm

        # Backpropagate through normalization
        grad_out = grad_out_norm / (norms + 1e-8) - (out_norm * (grad_out_norm * out).sum(dim=1, keepdim=True) / ((norms + 1e-8)**2))
    
        # Backpropagate through tanh
        # grad_out *= (1 - torch.tanh(out)**2).view_as(grad_out)
        # grad_out *= (out > 0).float().view_as(grad_out)

        # reshape grad_out back to the shape of the convolutional layer's output
        grad_out = grad_out.view(batch_size, out_channels, height, width)


        # Compute the gradient with respect to the convolutional weights
        conv_weight_grad = torch.nn.grad.conv2d_weight(
            self.x, self.weight.shape, grad_out, self.stride, self.F_padding[0], self.dilation, self.groups
        ) / batch_size

        # print('grad norm: ', conv_weight_grad.norm())
        # print('weight norm: ', self.weight.norm())

        return - conv_weight_grad 
    
    def update_rule_sample(self, target):
        # Reshape the output to [batch_size, output_channels * height * width]
        batch_size, out_channels, height, width = self.out.shape
        out = self.out.view(batch_size, -1) # [batch_size, height * width * output_channels]
        out = torch.tanh(out) # apply tanh -> still to understand why with this it works better
       
        norms = torch.norm(out, dim=1, p=2, keepdim=True) + 1e-8
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
        # loss = (positive_mask * (1 - similarity) + negative_mask * similarity.abs()).sum()

        # Add label smoothing to positive and negative masks
        if self.label_smoothing is not None:
            positive_mask = (1 - self.label_smoothing) * positive_mask + self.label_smoothing / (self.out.shape[0] - 1)
            negative_mask = (1 - self.label_smoothing) * negative_mask + self.label_smoothing / (self.out.shape[0] - 1)
            positive_mask.fill_diagonal_(0)
            negative_mask.fill_diagonal_(0)

        # Compute the gradient of the loss w.r.t. similarity
        # grad_similarity = 2 * (negative_mask - positive_mask) * similarity.abs()
        grad_similarity = negative_mask * similarity.sign() - positive_mask 
        grad_similarity /= self.out.shape[0]

        # Compute gradient of similarity w.r.t. normalized output (out_norm)
        grad_out = grad_similarity @ out_norm

        # Backpropagate through normalization
        grad_out = (grad_out * norms - out_norm * torch.sum(out * grad_out, dim=1, keepdim=True)) / norms**2
        # grad_out = grad_out_norm / norms ** 2
    
        # Backpropagate through tanh
        grad_out *= (1 - torch.tanh(out)**2).view_as(grad_out)
        # grad_out *= (out > 0).float().view_as(grad_out)

        # reshape grad_out back to the shape of the convolutional layer's output
        grad_out = grad_out.view(batch_size, out_channels, height, width)

        # Compute the gradient with respect to the convolutional weights
        conv_weight_grad = torch.nn.grad.conv2d_weight(
            self.x, self.weight.shape, grad_out, self.stride, self.F_padding[0], self.dilation, self.groups
        )
        return - conv_weight_grad 
    
    def regularize_orthogonality(self):
        weight = self.weight.view(self.weight.shape[0], -1)
        weight_norm = torch.linalg.norm(weight, dim=1, ord=2)
        weight = weight / weight_norm[:, None]
        similarity = weight @ weight.T
        similarity = similarity - torch.diag(torch.diag(similarity))

        dw_ortho = 2 * similarity.abs() @ weight
        dw_ortho = dw_ortho.view(self.weight.shape)
        return - dw_ortho.detach() 


    def update_rule_sample_old(self, target):
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
        grad_similarity = negative_mask * similarity.sign() - positive_mask
        grad_similarity /= self.out.shape[0]

        # Compute the gradient of the loss w.r.t. out
        out_norm_matrix = (out_norm[:, None] * out_norm[None, :]) # [batch_size, batch_size]

        grad_similarity_normalized = grad_similarity / out_norm_matrix # [batch_size, batch_size]
        grad_out = grad_similarity_normalized @ out


        # Reshape grad_out back to the shape of the convolutional layer's output
        grad_out = grad_out.view(batch_size, out_channels, height, width)


        # Compute the gradient with respect to the convolutional weights
        conv_weight_grad = torch.nn.grad.conv2d_weight(
            self.x, self.weight.shape, grad_out, self.stride, self.F_padding[0], self.dilation, self.groups
        )

        print('grad norm: ', conv_weight_grad.norm())

        return - conv_weight_grad 
        