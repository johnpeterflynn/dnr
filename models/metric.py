import torch

def mse_rgb32(output, target):
    criterion = torch.nn.MSELoss()
    loss = criterion(output, target)

    # Convert output, target from [-1,1] to [0,255]
    loss_rescaled = loss * (255 / 2) ** 2

    return loss_rescaled.item()

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
