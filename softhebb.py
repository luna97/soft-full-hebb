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
from sklearn.metrics import f1_score
from model import DeepSoftHebb, CONV, LINEAR, LinearSofHebb, POOL_MAX, POOL_AVG, POOL_ORIG
import argparse
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import wandb
import os
from datasets import CIFAR10, MNIST, IMAGENET, get_datasets, STL10
from utils import CustomStepLR
from utils import CLIP, L2NORM, L1NORM, MAXNORM, NONORM, DECAY, RELU, TANH, TRIANGLE
from conv import SOFTHEBB, ANTIHEBB, CHANNEL, SAMPLE, CHSAMPLE

# optimizers
SGD = 'sgd'
ADAMW = 'adamw'
MOMENTUM = 'momentum'


available_datasets = [CIFAR10, MNIST, IMAGENET, STL10]
available_normalizations = [L1NORM, L2NORM, MAXNORM, CLIP, NONORM, DECAY]
available_optimizers = [SGD, ADAMW, MOMENTUM]
available_conv_rules = [SOFTHEBB, ANTIHEBB, CHANNEL, SAMPLE, CHSAMPLE]
available_pooling = [POOL_MAX, POOL_AVG, POOL_ORIG]
available_activations = [RELU, TANH, TRIANGLE]

device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")

# Parse command line arguments
parser = argparse.ArgumentParser(description='SoftHebb Training')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.003, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--log', action='store_true', help='enable logging with wandb')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--save_model', action='store_true', help='save model')
# parser.add_argument('--augment_data', action='store_true', help='use data augmentation')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use (cifar10, mnist, or imagenet)')
parser.add_argument('--neuron_centric', action='store_true', help='use neuron-centric learning')
parser.add_argument('--learn_t', action='store_true', help='learn temperature')
parser.add_argument('--normalization', type=str, default="clip")
parser.add_argument('--two_step', action='store_true', help='use two steps learning')
parser.add_argument('--linear_head', action='store_true', help='use linear head')
parser.add_argument('--net_type', type=str, default=CONV, help='network type')
parser.add_argument('--device', type=str, default=device, help='device to use')
parser.add_argument('--optimizer', type=str, default=ADAMW, help='optimizer to use')
parser.add_argument('--initial_lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--no_momentum', action='store_true', help='use momentum')
parser.add_argument('--conv_rule', type=str, default=SAMPLE, help='convolution rule')
parser.add_argument('--regularize_orth', action='store_true', help='regularize orthogonal weights')
parser.add_argument('--conv_channels', type=int, default=96, help='number of input channels for convnet')
parser.add_argument('--conv_factor', type=int, default=4, help='factor size for convnet')
parser.add_argument('--use_batch_norm', action='store_true', help='use batch normalization')
parser.add_argument('--label_smoothing', type=float, default=None, help='label smoothing factor')
parser.add_argument('--offline', action='store_true', help='offline training')
parser.add_argument('--pooling_type', type=str, default=POOL_ORIG, help='pooling type')
parser.add_argument('--activation', type=str, default=TANH, help='activation function')
parser.add_argument('--full', action='store_true', help='use full network architecture')
parser.add_argument('--deterministic', action='store_true', help='set random seed')
parser.add_argument('--run_name', type=str, default='default', help='run name')
parser.add_argument('--linear_initial_lr', type=float, default=None, help='initial learning rate for linear model')
parser.add_argument('--linear_norm_type', type=str, default=None, help='normalization type for linear model')
parser.add_argument('--conv_depth', type=int, default=3, help='depth of the convolutional network')
args = parser.parse_args()

if args.deterministic:
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

device = args.device

if args.dataset.lower() not in available_datasets:
    raise ValueError(f"Dataset {args.dataset} not available. Choose one of {available_datasets}")
dataset = args.dataset.lower()

if args.net_type.lower() not in [CONV, LINEAR]:
    raise ValueError(f"Network type {args.net_type} not available. Choose one of {CONV}, {LINEAR}")

if args.normalization.lower() not in available_normalizations:
    raise ValueError(f"Normalization {args.normalization} not available. Choose one of {available_normalizations}")
normalization = args.normalization.lower()

if args.optimizer.lower() not in available_optimizers:
    raise ValueError(f"Optimizer {args.optimizer} not available. Choose one of {available_optimizers}")

if args.conv_rule.lower() not in available_conv_rules:
    raise ValueError(f"Convolution rule {args.conv_rule} not available. Choose one of {available_conv_rules}")

if args.pooling_type.lower() not in available_pooling:
    raise ValueError(f"Pooling type {args.pooling_type} not available. Choose one of {available_pooling}")

if args.activation.lower() not in available_activations:
    raise ValueError(f"Activation function {args.activation} not available. Choose one of {available_activations}")

if __name__ == "__main__":
    if args.log:
        wandb.init(
            project=f"softhebb-{dataset}-{args.net_type}",
            config=vars(args),
            mode='online' if not args.offline else 'offline',
            name=args.run_name
        )
        model_name = f'{wandb.run.id}.pth'
    else:
        model_name = "best_model.pth"
        
    in_channels = 3 if dataset == CIFAR10 or dataset == IMAGENET or dataset == STL10 else 1
    input_size = 32 if dataset == CIFAR10 else 28 if dataset == MNIST else 224 if dataset == IMAGENET else 96
    if args.net_type == LINEAR:
        model = LinearSofHebb(
            in_channels=in_channels,
            norm_type=normalization,
            two_steps=args.two_step,
            device=device,
            dropout=args.dropout,
            input_size=input_size,
            initial_lr=args.initial_lr,
            use_momentum=not args.no_momentum,
            use_batch_norm=args.use_batch_norm,
            label_smoothing=args.label_smoothing,
            activation=args.activation.lower()
        ).to(device)
    else:
        linear_initial_lr = args.linear_initial_lr if args.linear_initial_lr is not None else args.initial_lr
        linear_norm_type = args.linear_norm_type if args.linear_norm_type is not None else normalization
        model = DeepSoftHebb(
            device=device,
            in_channels=in_channels, 
            dropout=args.dropout, 
            input_size=input_size, 
            neuron_centric=args.neuron_centric,
            learn_t_invert=args.learn_t,
            norm_type=normalization,
            two_steps=args.two_step,
            linear_head=args.linear_head,
            initial_lr=args.initial_lr,
            use_momentum=not args.no_momentum,
            conv_rule=args.conv_rule,
            regularize_orth=args.regularize_orth,
            conv_channels=args.conv_channels,
            conv_factor=args.conv_factor,
            use_batch_norm=args.use_batch_norm,
            label_smoothing=args.label_smoothing,
            pooling=args.pooling_type,
            activation=args.activation.lower(),
            full=args.full,
            linear_initial_lr=linear_initial_lr,
            linear_norm_type=linear_norm_type,
            conv_depth=args.conv_depth
        ).to(device)

    model.train()

    params_to_optimize = [param for _, param in model.named_parameters() if param.requires_grad] if args.neuron_centric else model.classifier.parameters()


    if args.optimizer.lower() == ADAMW:
        optimizer = AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == MOMENTUM:
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    # print(f'optimizing {[name for name, param in params_to_optimize if param.requires_grad]}')

    # add lr scheduler
    if not args.neuron_centric:
        scheduler = CustomStepLR(optimizer, nb_epochs=50)

    print("Parameters that requires grad: ", [name for name, param in model.named_parameters() if param.requires_grad])
    criterion = nn.CrossEntropyLoss(reduction='mean')

    dataset_base, train_dataset, val_dataset, test_dataset = get_datasets(dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 

    print(f"Training on {dataset} with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples")

    total = len(train_dataloader)
    best_val_f1 = 0.0

    epoch_pbar = tqdm(range(args.epochs))
    for e in epoch_pbar:
        model.train()
        running_loss = 0.
        train_correct = []
        train_targets = []
        correct = 0
        train_total = 0
        # print memory requirement for the model
        # print(f"Memory requirement for the model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        pbar = tqdm(enumerate(train_dataloader, 0), total=total)
        for i, data in pbar:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            if args.neuron_centric:
                outputs = model(inputs, targets)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, targets)

            if loss.grad_fn is not None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if not args.two_step:
                model.step(targets)

            running_loss += loss.item()
            train_total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_targets.append(targets.detach().cpu())
            train_correct.append(predicted.detach().cpu())
            correct += (predicted == targets).sum().item()
            pbar.set_description(f"Train Loss: {running_loss / (i + 1):.4f}")

        if not args.neuron_centric:
            scheduler.step()
        acc_train = correct / train_total
        f1_train = f1_score(torch.cat(train_targets), torch.cat(train_correct), average='macro')

        # print(f"Memory requirement for the model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


        # Validation loop
        model.eval()
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

                outputs = model(inputs)

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
            torch.save(model, f"data/{model_name}")


        if args.log:
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
    if args.log:
        wandb.log({"Test Accuracy": acc_test, "Test F1": f1_test})

# remove the model 
os.remove(f"data/{model_name}")
