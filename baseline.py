import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import get_datasets   
import tqdm

# Define transformations for the training set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

dataset_base, train_dataset, val_dataset, test_dataset = get_datasets('mnist')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, input_size=28):
        super(CNN, self).__init__()
        out_channels_1 =16
        out_channels_2 = 16*4
        out_channels_3 = 16*8

        self.conv1 = nn.Conv2d(1, out_channels_1, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        out_size = (input_size - 2) // 2 + 1

        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels_1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        out_size = (out_size - 2) // 2 + 1

        self.conv3 = nn.Conv2d(out_channels_2, out_channels_3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels_2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        out_size = (out_size - 2) // 2 + 1

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear((out_size ** 2) * out_channels_3, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(self.bn1(x))))
        x = self.pool2(torch.relu(self.conv2(self.bn2(x))))
        x = self.pool3(torch.relu(self.conv3(self.bn3(x))))

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNN(input_size=28)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (inputs, labels) in pbar:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Log running loss
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], loss: {running_loss / (i + 1):.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_dataloader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    # Print training and validation loss
    # print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}')
    # print accuracy
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {100 * val_correct / val_total:.2f}%')

print('Finished Training')

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
