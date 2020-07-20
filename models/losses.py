import torch


# TODO: IS the photometric reproduction loss really just MAE?
def photometric_reproduction_loss(output, target):
    criterion = torch.nn.L1Loss()
    return criterion(output, target)


def laplacian_pyramid_l2_regularization(weights, lam):
    num_layers = len(weights)
    loss = 0
    for i, layer in enumerate(weights):
        lam_prime = lam * ((4 ** (num_layers - i - 1)) - 1)
        loss = loss + 0.5 * lam_prime * torch.norm(layer, 2)

    return loss


def laplacian_pyramid_l2_color_channel_regularization(weights, target, lam):
    num_layers = len(weights)
    avg_colors = target.sum((2, 3), keepdim=True)
    loss = 0
    for i, layer in enumerate(weights):
        lam_prime = lam * ((4 ** (num_layers - i - 1)) - 1)
        loss = loss + 0.5 * lam_prime * (torch.norm((layer[:, 0:3, :, :] - avg_colors), 2)
                                         + torch.norm(layer[:, 3:, :, :], 2))

    return loss


def photometric_rep_loss_laplacian_l2(output, target, weights, lam):
    return photometric_reproduction_loss(output, target)\
           + laplacian_pyramid_l2_regularization(weights, lam)


def photometric_rep_loss_laplacian_l2_color(output, target, weights, lam):
    return photometric_reproduction_loss(output, target)\
           + laplacian_pyramid_l2_color_channel_regularization(weights, target, lam)


def color_channel_regularization(weights, target, lam):
    avg_colors = target.sum((2, 3), keepdim=True)
    loss = 0
    for i, layer in enumerate(weights):
        loss = loss + torch.sum(torch.abs(layer[:, 0:3, :, :] - avg_colors))

    loss = lam * loss

    return loss
