import torch

# TODO: IS the photometric reproduction loss really just MAE?
def photometric_reproduction_loss(output, target):
    criterion = torch.nn.L1Loss()
    return criterion(output, target)
