import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

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

    # print(f"Accuracy: {acc_test:.4f}, F1: {f1_test:.4f}")
    return acc, f1, test_loss / total_test