import torch
from torch.nn.modules.utils import _pair
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import normalize_weights, CLIP
from conv import SoftHebbConv2d
from linear import SoftHebbLinear
  

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
            linear_head=False
    ):
        super(DeepSoftHebb, self).__init__()
        # block 1
        lol = 96
        self.bn1 = nn.BatchNorm2d(in_channels, affine=False).requires_grad_(False)
        out_channels_1 = lol
        out_channels_2 = out_channels_1 * 4
        out_channels_3 = out_channels_2 * 4
        self.conv1 = SoftHebbConv2d(
            in_channels=in_channels, 
            out_channels=out_channels_1, 
            kernel_size=5, 
            padding=2, 
            device=device, 
            initial_lr=0.08, # only for original version
            first_layer=True,
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert,
            norm_type=norm_type,
            two_steps=two_steps
        )
        self.activ1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        out_size = (input_size - 2) // 2 + 1

        # block 2
        self.bn2 = nn.BatchNorm2d(out_channels_1, affine=False).requires_grad_(False)
        self.conv2 = SoftHebbConv2d(
            in_channels=out_channels_1, 
            out_channels=out_channels_2,
            kernel_size=3, 
            device=device, 
            initial_lr=0.005, # only for original version
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert,
            norm_type=norm_type,
            two_steps=two_steps
        )
        self.activ2 = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        out_size = (out_size - 2) // 2 + 1

        # block 3
        self.bn3 = nn.BatchNorm2d(out_channels_2, affine=False).requires_grad_(False)
        self.conv3 = SoftHebbConv2d(
            in_channels=out_channels_2, 
            out_channels=out_channels_3,
            kernel_size=3, 
            device=device,
            initial_lr=0.01, # only for original version
            neuron_centric=neuron_centric and not unsupervised_first,
            learn_t_invert=learn_t_invert,
            norm_type=norm_type,
            two_steps=two_steps
        )
        self.activ3 = nn.ReLU() #Triangle(power=1.) #
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        out_size = (out_size - 2) // 2 + 1
        out_dim = (out_size ** 2) * out_channels_3

        # block 4
        self.flatten = nn.Flatten()
        self.bn_out = nn.BatchNorm1d(out_dim, affine=False).requires_grad_(False)


        if neuron_centric and not linear_head:
            self.classifier = SoftHebbLinear(out_dim, 10, device=device, two_steps=two_steps)
        else:
            self.classifier = nn.Linear(out_dim, 10)
            # self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, (input_size // 2) * 1536)

        self.neuron_centric = neuron_centric
        self.unsupervised_first = unsupervised_first
        self.linear_head = linear_head
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
        out = self.flatten(out)
        if self.neuron_centric and not self.linear_head:
            return self.classifier(self.dropout(self.bn_out(out)), target)
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
        self.conv1.step()
        self.conv2.step()
        self.conv3.step()
        if self.neuron_centric and not self.linear_head:
            self.classifier.step(target)

