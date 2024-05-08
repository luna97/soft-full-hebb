from ff import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

torch.manual_seed(42)
torch.mps.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.GaussianBlur(5, sigma=(1, 1.)),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 32
device = "cpu" #"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# datasets
trn_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
tain_dataset, val_dataset = torch.utils.data.random_split(trn_dataset, [50000, 10000])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_dataloader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ff) 
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ff) 

net = Net([794, 500, 500, 318], num_classes=10, device=device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0001) #, momentum=0.9, nesterov=True)

epoch_pbar = tqdm(range(100))
# temperature = 1
for e in epoch_pbar:
    batch_pbar = tqdm(train_dataloader, leave=False)
    losses = []
    for i, batch in enumerate(batch_pbar):
        # print(batch)
        x_pos = batch['pos'].to(device)
        x_neg = batch['neg'].to(device)
        out_pos, out_neg, loss_list = net(x_pos, x_neg)
        loss = loss_list.sum()
        losses.append(loss.item())
        batch_pbar.set_description(f'loss: {loss.item()}')
        if loss.grad_fn is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate 
    f1, acc = net.evaluate(val_dataloader)
    epoch_pbar.set_description(f'loss: {sum(losses) / len(losses)} - F1: {f1} - Acc: {acc}')
    # epoch_pbar.set_description(f'loss: {sum(losses) / len(losses)}')

f1, acc = net.evaluate(test_dataloader)
print(f"Accuracy: {acc} - F1: {f1}")

