import torch
import torch
import torchvision
from tqdm import tqdm
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import get_datasets, CIFAR10, MNIST, IMAGENET, STL10
import argparse

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--log', action='store_true', help='Enable logging with wandb')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
parser.add_argument('--net_type', type=str, default='linear', help='Type of network')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--no_momentum', action='store_true', help='Disable momentum in optimizer')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--run_name', type=str, default='default', help='run name')

args = parser.parse_args()

dataset = args.dataset.lower()
in_channels = 3 if dataset == CIFAR10 or dataset == IMAGENET or dataset == STL10 else 1
input_size = 32 if dataset == CIFAR10 else 28 if dataset == MNIST else 224 if dataset == IMAGENET else 96

# Define the Convolutional Neural Network
class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        lol = 32
        out_channels_1 = lol
        out_channels_2 = out_channels_1 * 2
        out_channels_3 = out_channels_2 * 2
        k_size_1, k_size_2, k_size_3 = 6, 4, 2
        stride_1, stride_2, stride_3 = 2, 2, 2
        padding_1, padding_2, padding_3 = 0, 0, 0

        pool_k_size_1, pool_k_size_2, pool_k_size_3 = 4, 4, 2
        pool_stride_1, pool_stride_2, pool_stride_3 = 2, 2, 2
        pool_padding_1, pool_padding_2, pool_padding_3 = 1, 1, 0

        self.bn1 = nn.BatchNorm2d(in_channels, affine=False).requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, k_size_1, stride_1, padding_1)
        out_size = ((32 - k_size_1 + 2 * padding_1) // stride_1) + 1
        # out_size = ((out_size - pool_k_size_1 + 2 * pool_padding_1) // pool_stride_1) + 1
        #self.pool1 = nn.MaxPool2d(kernel_size=pool_k_size_1, stride=pool_stride_1, padding=pool_padding_1)

        self.bn2 = nn.BatchNorm2d(out_channels_1, affine=False).requires_grad_(False)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, k_size_2, stride_2, padding_2)
        out_size = (out_size - k_size_2 + 2 * padding_2) // stride_2 + 1
       # out_size = (out_size - pool_k_size_2 + 2 * pool_padding_2) // pool_stride_2 + 1
        # self.pool2 = nn.MaxPool2d(kernel_size=pool_k_size_2, stride=pool_stride_2, padding=pool_padding_2)

        self.bn3 = nn.BatchNorm2d(out_channels_2, affine=False).requires_grad_(False)
        self.conv3 = nn.Conv2d(out_channels_2, out_channels_3, k_size_3, stride_3, padding_3)
        out_size = (out_size - k_size_3 + 2 * padding_3) // stride_3 + 1
       #  out_size = (out_size - pool_k_size_3 + 2 * pool_padding_3) // pool_stride_3 + 1
        # self.pool3 = nn.MaxPool2d(kernel_size=pool_k_size_3, stride=pool_stride_3, padding=pool_padding_3)
        
        out_dim = (out_size ** 2) * out_channels_3
        print(out_dim)
        self.bn_out = nn.BatchNorm1d(out_dim, affine=False).requires_grad_(False)
        self.fc1 = nn.Linear(out_dim, out_dim // 4)
        self.fc2 = nn.Linear(out_dim // 4, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.bn_out(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NetLinear(nn.Module):
    def __init__(self):
        super(NetLinear, self).__init__()
        self.fc1 = nn.Linear((input_size ** 2) * in_channels, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 2000)
        self.fc4 = nn.Linear(2000, 10)
        self.relu = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

batch_size = args.batch_size
log = args.log
dataset = args.dataset
net_type = args.net_type
lr = args.lr
no_momentum = args.no_momentum
device = args.device
epochs = args.epochs
dataset_base, train_dataset, val_dataset, test_dataset = get_datasets(dataset)

best_val_f1 = 0.0

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

# Initialize the network and optimizer
net = NetLinear().to(device)
criterion = nn.CrossEntropyLoss(reduction='mean')
if no_momentum:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0)
else:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# Initialize wandb
if log: 
    wandb.init(
        project=f"softhebb-{dataset}-{net_type}",
        config=vars(args),
        name=args.run_name
    )
    model_name = f'{wandb.run.id}.pth'
else:
    model_name = "best_model_normal.pth"

epoch_pbar = tqdm(range(epochs))
total = len(train_dataloader)

# Training loop
for e in epoch_pbar:  # Change the number of epochs as needed
    net.train()
    running_loss = 0.
    train_correct = []
    train_targets = []
    correct = 0
    train_total = 0

    pbar = tqdm(enumerate(train_dataloader, 0), total=total)

    for i, data in pbar:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        # count non-zero gradients
        # print(f"fc grad: {model.classifier.Ci.grad.abs().sum()}")
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        train_total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_targets.append(targets.detach().cpu())
        train_correct.append(predicted.detach().cpu())
        correct += (predicted == targets).sum().item()
        pbar.set_description(f"Train Loss: {running_loss / (i + 1):.4f}")
    
    acc_train = correct / train_total
    f1_train = f1_score(torch.cat(train_targets), torch.cat(train_correct), average='macro')

    # Validation loop
    net.eval()
    val_loss = 0.0
    val_correct = []
    val_targets = []
    correct = 0
    pbar_val = tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader), leave=False)
    total_val = 0

    with torch.no_grad():
        for i, data in pbar_val:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            total_val += targets.size(0)

            _, predicted = torch.max(outputs.data, 1)
            val_targets.append(targets.detach().cpu())
            val_correct.append(predicted.detach().cpu())
            correct += (predicted == targets).sum().item()  
            
    acc = correct / total_val
    f1 = f1_score(torch.cat(val_targets), torch.cat(val_correct), average='macro')
    print(f"Train Loss: {running_loss / total:.4f}, Train Acc: {acc_train:.4f}, Train F1: {f1_train:.4f}")
    print(f"Val Loss: {val_loss / len(val_dataloader):.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

    if f1 >= best_val_f1:
        best_val_f1 = f1
        # best_model = model.save()
        print(f"Best model at epoch: {e}")
        # if args.save_model:
        torch.save(net, f"data/{model_name}.pth")

    if log:
        wandb.log({
            "Validation Loss": val_loss / len(val_dataloader),
            "Validation Accuracy": acc, 
            "Validation F1 score": f1,
            "Training Loss": running_loss / total,
            "Training Accuracy": acc_train,
            "Training F1 score": f1_train
        })

print(f"loading best model for testing")
model = torch.load(f"data/{model_name}.pth")

# Test loop
model.eval()
test_loss = 0.0
test_res = []
test_targets = []
total_test = 0
correct = 0 
pbar_test = tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader))

with torch.no_grad():
    for i, data in pbar_test:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        test_targets.append(targets.detach().cpu())
        test_res.append(predicted.detach().cpu())
        correct += (predicted == targets).sum().item()
        total_test += targets.size(0)

acc_test = correct / total_test
f1_test = f1_score(torch.cat(test_targets), torch.cat(test_res), average='macro')

print(f"Test Accuracy: {acc_test:.4f}, Test F1: {f1_test:.4f}")

# Log final metrics to wandb
if log:
    wandb.log({"Test Accuracy": acc_test, "Test F1": f1_test})
