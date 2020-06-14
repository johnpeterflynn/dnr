import torch

# TODO: IS the photometric reproduction loss really just MAE?
def photometric_reproduction_loss(output, targer):
    return torch.nn.L1Loss()
