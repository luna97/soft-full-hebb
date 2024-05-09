"""
Demo single-file script to train a ConvNet on CIFAR10 using SoftHebb, an unsupervised, efficient and bio-plausible
learning algorithm
"""
import math
import warnings
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
import torchvision
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from model import DeepSoftHebb
import argparse
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import wandb

torch.manual_seed(42)
torch.cuda.manual_seed(42)


CIFAR10 = 'cifar10'
MNIST = 'mnist'
IMAGENET = 'imagenet'
available_datasets = [CIFAR10, MNIST, IMAGENET]

# Parse command line arguments
parser = argparse.ArgumentParser(description='SoftHebb Training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--log', action='store_true', help='enable logging with wandb')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--save_model', action='store_true', help='save model')
parser.add_argument('--augment_data', action='store_true', help='use data augmentation')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use (cifar10, mnist, or imagenet)')
parser.add_argument('--use_neuron_centric', action='store_true', help='use neuron-centric learning')
args = parser.parse_args()

print("Using neuron-centric learning" if args.use_neuron_centric else "Using softhebb original learning")

if args.dataset.lower() not in available_datasets:
    raise ValueError(f"Dataset {args.dataset} not available. Choose one of {available_datasets}")
dataset = args.dataset.lower()

norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if dataset == CIFAR10 else transforms.Normalize((0.1307,), (0.3081,))
transform_train = transforms.Compose([
    transforms.ToTensor(),
    norm_transform
])

if args.augment_data and dataset == CIFAR10:
    augmentation = transforms.Compose([ 
        transforms.RandomPosterize(bits=5, p=0.2),
        transforms.RandomAdjustSharpness(2, p=0.2),
        # transforms.RandomEqualize(0.2),
        transforms.RandomHorizontalFlip(0.2)
    ])
    transform_train.transforms.insert(0, augmentation)

transform = transforms.Compose([
    transforms.ToTensor(),
    norm_transform
])

class FastCIFAR10(torchvision.datasets.CIFAR10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)

        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)
        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.data[index]
        target = self.targets[index]

        return img, target

# Main training loop CIFAR10
if __name__ == "__main__":
    if args.log: wandb.init(
        project="softhebb-cifar10",
        config=vars(args),
    )
        
    device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")
    in_channels = 3 if dataset == CIFAR10 or dataset == IMAGENET else 1
    input_size = 32 if dataset == CIFAR10 else 28 if dataset == MNIST else 224
    model = DeepSoftHebb(device=device, in_channels=in_channels, dropout=args.dropout, input_size=input_size, use_neuron_centric=args.use_neuron_centric)
    model.to(device)

    optimizer = Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # add lr scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print([name for name, param in model.named_parameters() if param.requires_grad])
    criterion = nn.CrossEntropyLoss()

    if dataset == CIFAR10:
        dataset_base = FastCIFAR10(root='./data', train=True, download=True)
        # dataset_base = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = FastCIFAR10(root='./data', train=False, download=True)
        # test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    elif dataset == MNIST:
        dataset_base = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif dataset == IMAGENET:
        dataset_base = datasets.ImageNet(root='./data', train=True, transform=transform_train, download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = datasets.ImageNet(root='./data', train=False, transform=transform, download=True)

    train_dataloader = DataLoader(dataset_base, batch_size=args.batch_size, shuffle=True) 
    val_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 

    # Unsupervised training with SoftHebb
    total = len(train_dataloader)

    best_model = None
    best_val_acc = 0.0

    # show the full dataset to the model before training
    if not args.use_neuron_centric:
        print("Unsupervised hebbian learning.. ")
        # full_dataloader = DataLoader(dataset_base, batch_size=10, shuffle=True)
        # model.unsupervised_train()
        # with torch.no_grad():
        #   for data in tqdm(full_dataloader):
        #        inputs, _ = data
        #        inputs = inputs.to(device)
        #        _ = model(inputs)
        # model.unsupervised_eval()
        # load weights
        model.conv1.weight.data = torch.load("conv1_weight.pth")
        model.conv2.weight.data = torch.load("conv2_weight.pth")
        model.conv3.weight.data = torch.load("conv3_weight.pth")


    epoch_pbar = tqdm(range(args.epochs))
    for e in epoch_pbar:
        model.train()
        running_loss = 0.
        train_correct = []
        train_targets = []
        pbar = tqdm(enumerate(train_dataloader, 0), total=total)
        for i, data in pbar:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            #if loss.grad_fn is not None:
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_targets.append(targets.detach().cpu())
            train_correct.append(predicted.detach().cpu())
            pbar.set_description(f"Train Loss: {running_loss / (i + 1):.4f}")

        scheduler.step()
        acc_train = (torch.cat(train_correct) == torch.cat(train_targets)).sum().item() / len(train_dataset)
        f1_train = f1_score(torch.cat(train_targets), torch.cat(train_correct), average='macro')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = []
        val_targets = []
        total_val = len(val_dataloader)
        pbar_val = tqdm(enumerate(val_dataloader, 0), total=total_val, leave=False)

        with torch.no_grad():
            for i, data in pbar_val:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_targets.append(targets.detach().cpu())
                val_correct.append(predicted.detach().cpu())
                
        acc = accuracy_score(torch.cat(val_targets), torch.cat(val_correct))
        f1 = f1_score(torch.cat(val_targets), torch.cat(val_correct), average='macro')
        print(f"Train Loss: {running_loss / total:.4f}, Train Acc: {acc_train:.4f}, Train F1: {f1_train:.4f}")
        print(f"Val Loss: {val_loss / total_val:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

        if best_val_acc < acc:
            best_val_acc = acc
            best_model = model.state_dict()
            if args.save_model:
                torch.save(best_model, "best_model.pth")

        if args.log:
            wandb.log({
                "Validation Loss": val_loss / total_val,
                "Validation Accuracy": acc, 
                "Validation F1 score": f1,
                "Training Loss": running_loss / total,
                "Training Accuracy": acc_train,
                "Training F1 score": f1_train
            })

    model.load_state_dict(best_model)

    # Test loop
    model.eval()
    test_loss = 0.0
    test_res = []
    test_targets = []
    total_test = len(test_dataloader)
    pbar_test = tqdm(enumerate(test_dataloader, 0), total=total_test)

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

    acc_test = accuracy_score(torch.cat(test_targets), torch.cat(test_res))
    f1_test = f1_score(torch.cat(test_targets), torch.cat(test_res), average='macro')

    print(f"Test Accuracy: {acc_test:.4f}, Test F1: {f1_test:.4f}")

    # Log final metrics to wandb
    if args.log:
        wandb.log({"Test Accuracy": acc_test, "Test F1": f1_test})
