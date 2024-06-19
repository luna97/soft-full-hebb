
import torch
import torchvision
import torchvision.transforms as transforms
import math
from torchvision import datasets

CIFAR10 = 'cifar10'
MNIST = 'mnist'
IMAGENET = 'imagenet'
STL10 = 'stl10'

class FastCIFAR10(torchvision.datasets.CIFAR10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)

        data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)
        data = torch.movedim(data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        self.data = transforms.functional.normalize(data, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
    
class FastMNIST(torchvision.datasets.MNIST):
    """
    Improves performance of training on MNIST by removing the PIL interface and pre-loading on the GPU (2-3x speedup).
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)

        self.data = self.data.to(dtype=torch.float, device=device).div_(255)
        self.data = torch.movedim(self.data, -1, 1)
        self.data = transforms.functional.normalize(self.data, mean=(0.5,), std=(0.5,)).unsqueeze(1)
        self.targets = self.targets.to(device=device)

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

class FastSTL10(torchvision.datasets.STL10):
    """
    Improves performance of training on STL10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)

        data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)
        # data = torch.movedim(data, -1, 1)
        self.data = transforms.functional.normalize(data, mean=(0.507, 0.487, 0.441), std=(0.267, 0.256, 0.276))
        self.targets = torch.tensor(self.labels, device=device)

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



def get_datasets(dataset):
    """
    Get the dataset and split it into training, validation, and test sets.
    """
    if dataset == CIFAR10:
        # transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #])
        dataset_base = FastCIFAR10(root='./data', train=True, download=True)
        #dataset_base = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = FastCIFAR10(root='./data', train=False, download=True)
        #test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    elif dataset == MNIST:
        #transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.5], std=[0.5])
        #])
        dataset_base = FastMNIST(root='./data', train=True, download=True)
        # dataset_base = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = FastMNIST(root='./data', train=False, download=True)
        #test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif dataset == IMAGENET:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_base = datasets.ImageNet(root='./data', train=True, transform=transform, download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = datasets.ImageNet(root='./data', train=False, transform=transform, download=True)
    elif dataset == STL10:
        dataset_base = FastSTL10(root='./data', split='train', download=True)
        split = [ math.floor(0.9 * len(dataset_base)), math.ceil(0.1 * len(dataset_base)) ]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_base, split)
        test_dataset = FastSTL10(root='./data', split='test', download=True)
    else:
        raise ValueError("Unknown dataset")

    return dataset_base, train_dataset, val_dataset, test_dataset

