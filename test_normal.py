import torch
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

        self.bn1 = nn.BatchNorm2d(3, affine=False).requires_grad_(False)
        self.conv1 = nn.Conv2d(3, out_channels_1, k_size_1, stride_1, padding_1)
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

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize the network and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):  # Change the number of epochs as needed
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 99:  # Print the loss every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            print(f'Accuracy: {100 * correct / total:.2f}%')
            running_loss = 0.0

print('Finished training')