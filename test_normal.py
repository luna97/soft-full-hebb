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
from torch.optim import AdamW
from model import NetConv, NetLinear

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--log', action='store_true', help='Enable logging with wandb')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
parser.add_argument('--net_type', type=str, default='linear', help='Type of network')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--no_momentum', action='store_true', help='Disable momentum in optimizer')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--run_name', type=str, default='default', help='run name')
parser.add_argument('--offline', action='store_true', help='offline training')
parser.add_argument('--optimizer', type=str, default='momentum', help='optimizer to use')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--conv_factor', type=int, default=4, help='conv factor')

args = parser.parse_args()

dataset = args.dataset.lower()
in_channels = 3 if dataset == CIFAR10 or dataset == IMAGENET or dataset == STL10 else 1
input_size = 32 if dataset == CIFAR10 else 28 if dataset == MNIST else 224 if dataset == IMAGENET else 96

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
net = NetConv(conv_factor=args.conv_factor, dropout=args.dropout, in_channels=in_channels) if net_type == 'conv' else NetLinear(in_channels=in_channels, input_size=input_size)
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = AdamW(net.parameters(), lr=lr)
if args.optimizer == 'adamw':
    optimizer = AdamW(net.parameters(), lr=lr)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0)
else:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# Initialize wandb
if log: 
    wandb.init(
        project=f"softhebb-{dataset}-{net_type}",
        config=vars(args),
        name=args.run_name,
        mode='online' if not args.offline else 'offline',
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

    # print(f"Memory requirement for the model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # print which variables are requiring the most memory
    
    pbar = tqdm(enumerate(train_dataloader, 0), total=total)

    for i, data in pbar:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
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
        torch.save(net, f"data/{model_name}")

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
model = torch.load(f"data/{model_name}")

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
