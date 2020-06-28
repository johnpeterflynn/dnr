import torch

# TODO: IS the photometric reproduction loss really just MAE?
def photometric_reproduction_loss(output, target, ignore_edge_pixels=None):
    criterion = torch.nn.L1Loss(reduction='none')
    loss = criterion(output, target)

    if ignore_edge_pixels is not None:
        x, y = ignore_edge_pixels
        loss = loss[:, :, y:-y, x:-x]

    return loss.mean()
