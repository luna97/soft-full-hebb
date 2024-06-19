import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score


CLIP = "clip"
L2NORM = "l2norm"
L1NORM = "l1norm"
MAXNORM = "maxnorm"
NONORM = "nonorm"
DECAY = "decay"
SOFTMAX = "softmax"
TRIANGLE = "triangle"

RELU = "relu"
TANH = "tanh"

def normalize_weights(weights, norm_type, dim=0):
    if norm_type == CLIP:
        return weights.clip(-10, 10)
    elif norm_type == L2NORM:
        return F.normalize(weights, p=2, dim=dim)
    elif norm_type == L1NORM:
        return F.normalize(weights, p=1, dim=dim)
    elif norm_type == MAXNORM:
        return weights / (torch.abs(weights).amax() + 1e-30)
    elif norm_type == NONORM:
        return weights
    elif norm_type == DECAY:
        return weights * 0.9995
    elif norm_type == SOFTMAX:
        return F.softmax(weights, dim=dim)
    else:
        raise ValueError(f"Invalid norm type: {norm_type}")
    
def balanced_credit_fn(out):
    # one if positive, 0 if negative
    # return torch.where(out > 0, torch.ones_like(out), torch.zeros_like(out))
    return 1.0 - F.tanh(out) ** 2


def evaluate(model, loader, loss_fn, device):
    # Test loop
    model.eval()
    test_loss = 0.0
    test_res = []
    test_targets = []
    total_test = 0
    correct = 0 
    pbar_test = tqdm(enumerate(loader, 0), total=len(loader))

    with torch.no_grad():
        for i, data in pbar_test:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_targets.append(targets.detach().cpu())
            test_res.append(predicted.detach().cpu())
            correct += (predicted == targets).sum().item()
            total_test += targets.size(0)

    acc = correct / total_test
    f1 = f1_score(torch.cat(test_targets), torch.cat(test_res), average='macro')

    return acc, f1, test_loss / total_test
    

        
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
    

class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power