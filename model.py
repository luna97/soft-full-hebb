import torch
from torch.nn.modules.utils import _pair
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import normalize_weights, CLIP, L2NORM, L1NORM, MAXNORM, NONORM, DECAY, SOFTMAX, TANH, RELU
from conv import SoftHebbConv2d
from linear import SoftHebbLinear
from conv import SOFTHEBB, ANTIHEBB, CHANNEL, SAMPLE

CONV = 'conv'
LINEAR = 'linear'

POOL_MAX = 'max'
POOL_AVG = 'avg'
POOL_ORIG = 'orig'

class DeepSoftHebb(nn.Module):
    def __init__(
            self, 
            device = 'cpu', 
            in_channels=1, 
            dropout=0.0, 
            input_size=32, 
            neuron_centric=False, 
            unsupervised_first=False, 
            learn_t_invert=False, 
            norm_type=CLIP,
            two_steps=False,
            linear_head=False,
            initial_lr=0.001,
            use_momentum=False,
            conv_rule=SAMPLE,
            regularize_orth=False,
            conv_channels=96,
            conv_factor=4,
            use_batch_norm=False,
            label_smoothing=None,
            pooling=POOL_ORIG,
            activation=TANH,
            full=False,
    ):
        super(DeepSoftHebb, self).__init__()
        # block 1
        out_channels_1 = conv_channels
        out_channels_2 = out_channels_1 * conv_factor
        out_channels_3 = out_channels_2 * conv_factor
        k_size_1, k_size_2, k_size_3 = 5, 3, 3
        stride_1, stride_2, stride_3 = 1, 1, 1
        padding_1, padding_2, padding_3 = 2, 1, 1
        pool_k_size_1, pool_k_size_2, pool_k_size_3 = 4, 4, 2
        pool_stride_1, pool_stride_2, pool_stride_3 = 2, 2, 2
        pool_padding_1, pool_padding_2, pool_padding_3 = 1, 1, 0
        self.full=full

        if activation == RELU:
            self.act = nn.ReLU()
        elif activation == TANH:
            self.act = nn.Tanh()
        else:
            self.act = nn.Tanh()
            activation = TANH

        self.use_batch_norm = use_batch_norm

        if pooling == POOL_MAX:
            self.pool1 = nn.MaxPool2d(kernel_size=pool_k_size_1, stride=pool_stride_1, padding=pool_padding_1)
            self.pool2 = nn.MaxPool2d(kernel_size=pool_k_size_2, stride=pool_stride_2, padding=pool_padding_2)
            self.pool3 = nn.MaxPool2d(kernel_size=pool_k_size_3, stride=pool_stride_3, padding=pool_padding_3)
        elif pooling == POOL_AVG:
            self.pool1 = nn.AvgPool2d(kernel_size=pool_k_size_1, stride=pool_stride_1, padding=pool_padding_1)
            self.pool2 = nn.AvgPool2d(kernel_size=pool_k_size_2, stride=pool_stride_2, padding=pool_padding_2)
            self.pool3 = nn.AvgPool2d(kernel_size=pool_k_size_3, stride=pool_stride_3, padding=pool_padding_3)
        else:
            self.pool1 = nn.MaxPool2d(kernel_size=pool_k_size_1, stride=pool_stride_1, padding=pool_padding_1)
            self.pool2 = nn.MaxPool2d(kernel_size=pool_k_size_2, stride=pool_stride_2, padding=pool_padding_2)
            self.pool3 = nn.AvgPool2d(kernel_size=pool_k_size_3, stride=pool_stride_3, padding=pool_padding_3) 
        
        # block 1
        self.conv1 = SoftHebbConv2d(
            in_channels=in_channels, 
            out_channels=out_channels_1, 
            kernel_size=k_size_1, 
            stride=stride_1,
            padding=padding_1,
            device=device, 
            initial_lr=initial_lr,
            first_layer=True,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert,
            norm_type=norm_type,
            two_steps=two_steps,
            inp_size=input_size,
            use_momentum=use_momentum,
            conv_rule=conv_rule,
            regularize_orth=regularize_orth,
            label_smoothing=label_smoothing,
            activation=activation
        )
        out_size = ((input_size - k_size_1 + 2 * padding_1) // stride_1) + 1
        out_size = ((out_size - pool_k_size_1 + 2 * pool_padding_1) // pool_stride_1) + 1

        # block 2
        self.conv2 = SoftHebbConv2d(
            in_channels=out_channels_1, 
            out_channels=out_channels_2,
            kernel_size=k_size_2,
            stride=stride_2,
            device=device, 
            padding=padding_2,
            initial_lr=initial_lr,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert,
            norm_type=norm_type,
            two_steps=two_steps,
            inp_size=out_size,
            use_momentum=use_momentum,
            conv_rule=conv_rule,
            regularize_orth=regularize_orth,
            label_smoothing=label_smoothing,
            activation=activation
        )
        out_size = (out_size - k_size_2 + 2 * padding_2) // stride_2 + 1
        out_size = (out_size - pool_k_size_2 + 2 * pool_padding_2) // pool_stride_2 + 1

        # block 3
        self.conv3 = SoftHebbConv2d(
            in_channels=out_channels_2, 
            out_channels=out_channels_3,
            kernel_size=k_size_3,
            stride=stride_3,
            padding=padding_3,
            device=device,
            initial_lr=initial_lr,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert,
            norm_type=norm_type,
            two_steps=two_steps,
            inp_size=out_size,
            use_momentum=use_momentum,
            conv_rule=conv_rule,
            regularize_orth=regularize_orth,
            label_smoothing=label_smoothing,
            activation=activation
        )
        out_size = (out_size - k_size_3 + 2 * padding_3) // stride_3 + 1
        out_size = (out_size - pool_k_size_3 + 2 * pool_padding_3) // pool_stride_3 + 1
        out_dim = (out_size ** 2) * out_channels_3

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels, affine=False).requires_grad_(False)
            self.bn2 = nn.BatchNorm2d(out_channels_1, affine=False).requires_grad_(False)
            self.bn3 = nn.BatchNorm2d(out_channels_2, affine=False).requires_grad_(False)
            self.bn_out = nn.BatchNorm1d(out_dim, affine=False).requires_grad_(False)

        # block 4
        self.flatten = nn.Flatten()

        if neuron_centric and not linear_head:
            if full:
                self.fc1 = SoftHebbLinear(
                    out_dim, 
                    out_dim // 4,
                    device=device,
                    two_steps=two_steps, 
                    norm_type=norm_type,
                    last_layer=False,
                    initial_lr=initial_lr * 10,
                    use_momentum=use_momentum,
                    label_smoothing=label_smoothing,
                    activation=activation
                )
                self.fc2 = SoftHebbLinear(
                    in_channels=out_dim // 4, 
                    out_channels=10,
                    device=device, 
                    two_steps=two_steps, 
                    norm_type=norm_type,
                    last_layer=True,
                    initial_lr=initial_lr * 10,
                    use_momentum=use_momentum,
                    label_smoothing=label_smoothing,
                    activation=activation
                )
            else:
                self.fc2 = SoftHebbLinear(
                    in_channels=out_dim, 
                    out_channels=10,
                    device=device, 
                    two_steps=two_steps, 
                    norm_type=norm_type,
                    last_layer=True,
                    initial_lr=initial_lr,
                    use_momentum=use_momentum,
                    label_smoothing=label_smoothing,
                    activation=activation
                )
        else:
            self.classifier = nn.Linear(out_dim, 10)

        self.neuron_centric = neuron_centric
        self.unsupervised_first = unsupervised_first
        self.linear_head = linear_head
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, out, target=None):
        # block 1
        if self.use_batch_norm: out = self.bn1(out)
        out = self.act(self.conv1(out, target=target))
        out = self.pool1(out)
        # block 2
        if self.use_batch_norm: out = self.bn2(out)
        out = self.act(self.conv2(out, target=target))
        out = self.pool2(out)
        # block 3
        if self.use_batch_norm: out = self.bn3(out)
        out = self.act(self.conv3(out, target=target))
        out = self.pool3(out)

        # block 4
        out = self.flatten(out)
        if self.neuron_centric and not self.linear_head:
            if self.full:
                out = self.fc1(self.dropout(out), target=target)
                out = self.fc2(self.dropout(out), target=target)
            else:
                out = self.fc2(self.dropout(out), target=target)
            return out
        else:
            return self.classifier(self.dropout(out))
        
    def unsupervised_eval(self):
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()

    def step(self, target):
        self.conv1.step(target=target)
        self.conv2.step(target=target)
        self.conv3.step(target=target)
        # self.conv3.step(credit=self.fc2.credit)
        if self.neuron_centric and not self.linear_head:
            if self.full:
                self.fc1.step(target=target)
                self.fc2.step(target=target)
            else:
                self.fc2.step(target=target)

class LinearSofHebb(nn.Module):
    def __init__(
            self, 
            device = 'cpu', 
            in_channels=1, 
            dropout=0.0, 
            input_size=32, 
            norm_type=CLIP,
            two_steps=False,
            use_momentum=False,
            initial_lr=0.001,
            use_batch_norm=False,
            label_smoothing=None,
            activation=TANH
    ):
        super(LinearSofHebb, self).__init__()

        self.fc1 = SoftHebbLinear(
            in_channels=in_channels * (input_size ** 2), 
            out_channels=2000,
            device=device, 
            two_steps=two_steps, 
            norm_type=norm_type,
            initial_lr=initial_lr,
            use_momentum=use_momentum,
            label_smoothing=label_smoothing,
            activation=activation
        )
        # self.bn1 = nn.BatchNorm1d(512, affine=False).requires_grad_(False)

        self.fc2 = SoftHebbLinear(
            in_channels=2000, 
            out_channels=2000,
            device=device, 
            two_steps=two_steps, 
            norm_type=norm_type,
            initial_lr=initial_lr,
            use_momentum=use_momentum,
            label_smoothing=label_smoothing,
            activation=activation
        )
        #self.bn2 = nn.BatchNorm1d(512, affine=False).requires_grad_(False)
        self.fc3 = SoftHebbLinear(
            in_channels=2000, 
            out_channels=2000,
            device=device, 
            two_steps=two_steps, 
            norm_type=norm_type,
            initial_lr=initial_lr,
            use_momentum=use_momentum,
            label_smoothing=label_smoothing,
            activation=activation
        )
        #self.bn3 = nn.BatchNorm1d(128, affine=False).requires_grad_(False)
        self.fc4 = SoftHebbLinear(
            in_channels=2000, 
            out_channels=10,
            device=device, 
            two_steps=two_steps, 
            norm_type=norm_type,
            last_layer=True,
            initial_lr=initial_lr,
            use_momentum=use_momentum,
            label_smoothing=label_smoothing,
            activation=activation
        )

        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()
        self.use_batch_norm = use_batch_norm

    def forward(self, out, target=None):
        # flatten
        out = out.view(out.size(0), -1)
        out = self.fc1(self.dropout(out), target=target)
        out = self.act(out)
        if self.use_batch_norm: out = self.bn1(out)
        out = self.fc2(self.dropout(out), target=target)
        out = self.act(out)
        if self.use_batch_norm: out = self.bn2(out)
        out = self.fc3(self.dropout(out), target=target)
        out = self.act(out)
        if self.use_batch_norm: out = self.bn3(out)
        out = self.fc4(self.dropout(out), target=target)
        return out
    
    def step(self, target):
        self.fc1.step(target=target)
        self.fc2.step(target=target)
        self.fc3.step(target=target)
        self.fc4.step(target=target)


