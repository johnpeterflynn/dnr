import torch

# TODO: IS the photometric reproduction loss really just MAE?
def photometric_reproduction_loss(output, target):
    criterion = torch.nn.L1Loss()
    return criterion(output, target)


def laplacian_pyramid_l2_regularization(weights, lam):
    loss = 0
    num_layers = len(weights)
    for i, layer in enumerate(weights):
        lam_prime = lam * ((4 ** (num_layers - i - 1)) - 1)
        loss = loss + 0.5 * lam_prime * torch.norm(layer, 2)

    return loss


def photometric_rep_loss_laplacian_l2(output, target, weights, lam):
    return photometric_reproduction_loss(output, target)\
           + laplacian_pyramid_l2_regularization(weights, lam)
