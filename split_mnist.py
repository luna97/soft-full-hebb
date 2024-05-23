import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import wandb
from model import DeepSoftHebb, CLIP
from datasets import CIFAR10, MNIST, IMAGENET, get_datasets
from tqdm import tqdm
from sklearn.metrics import f1_score
from utility import evaluate

# set seed for reproducibility
torch.manual_seed(42)
torch.mps.manual_seed(42)

torch.autograd.set_detect_anomaly(True)

# Load the MNIST dataset

dataset = CIFAR10
dataset_base, train_dataset, val_dataset, test_dataset = get_datasets(dataset)

tasks = [
    [0, 1],  # Task 1: Digits 0 vs 1
    [2, 3],  # Task 2: Digits 2 vs 3
    [4, 5], 
    [6, 7],
    [8, 9] 
]


def create_single_head_split(dataset, task_classes):
    """Creates a subset for a single-headed task"""
    indices = torch.where(torch.logical_or(dataset.targets == task_classes[0], 
                                           dataset.targets == task_classes[1]))[0]
    # set all targets to 0 or 1
    targets = (dataset.targets[indices] == task_classes[1]).long()
    dataset.targets[indices] = targets
    return Subset(dataset, indices)

def split_train_val(dataset, val_size=0.15):
    """Splits a dataset into training and validation sets"""
        
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])


train_datasets = []
val_datasets = []
test_datasets = []

for task in tasks:
    train_dataset, val_dataset = split_train_val(create_single_head_split(dataset_base, task))
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)
    test_datasets.append(create_single_head_split(test_dataset, task))

batch_size = 64

train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) 
                    for dataset in train_datasets]
val_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False)
                     for dataset in val_datasets]
test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False) 
                   for dataset in test_datasets]


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
in_channels = 3 if dataset == CIFAR10 or dataset == IMAGENET else 1
input_size = 32 if dataset == CIFAR10 else 28 if dataset == MNIST else 224
lr = 1.
wd = 0.0001
model = DeepSoftHebb(
    device=device,
    in_channels=in_channels,
    dropout=0.1, 
    input_size=input_size,
    neuron_centric=True,
    unsupervised_first=False,
    learn_t_invert=False,
    norm_type=CLIP,
    two_steps=False,
    linear_head=False
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()


# print num parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
    
for i, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    print(f"\n------------------Training task {tasks[i]}------------------")

    epoch_pbar = tqdm(range(1))
    for e in epoch_pbar:
        model.train()

        running_loss = 0.
        train_correct = []
        train_targets = []
        correct = 0
        train_total = 0
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        for i, data in pbar:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs, targets)

            loss = loss_fn(outputs, targets)

            if loss.grad_fn is not None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.step(targets)    


            running_loss += loss.item()
            train_total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_targets.append(targets.detach().cpu())
            train_correct.append(predicted.detach().cpu())
            correct += (predicted == targets).sum().item()
            pbar.set_description(f"Train Loss: {running_loss / train_total:.4f}")

        acc_train = correct / train_total
        f1_train = f1_score(torch.cat(train_targets), torch.cat(train_correct), average='macro')

        val_acc, val_f1, val_loss = evaluate(model, val_loader, loss_fn, device)

        print(f"Train Loss: {running_loss / train_total:.4f}, Train Acc: {acc_train:.4f}, Train F1: {f1_train:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")


    test_acc, f1_test, test_loss = evaluate(model, test_loader, loss_fn, device)

    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {f1_test:.4f}, Test Loss: {test_loss:.4f}")


print("----Final test-----")
for i, test_loader in enumerate(test_loaders):
    test_acc, f1_test, test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Task {tasks[i]}: Test Accuracy: {test_acc:.4f}, Test F1: {f1_test:.4f}, Test Loss: {test_loss:.4f}")
