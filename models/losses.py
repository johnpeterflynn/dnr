import torch
from .vgg16 import VGG16


# TODO: IS the photometric reproduction loss really just MAE?
def photometric_reproduction_loss(output, target):
    criterion = torch.nn.L1Loss()
    return criterion(output, target)


def laplacian_pyramid_l2_regularization(weights, lam):
    num_layers = len(weights)
    loss = 0
    for i, layer in enumerate(weights):
        lam_prime = lam * ((2 ** (num_layers - i - 1)) - 1)
        loss = loss + 0.5 * lam_prime * torch.norm(layer, 2)

    return loss


def laplacian_pyramid_l2_color_channel_regularization(weights, target, lam):
    num_layers = len(weights)
    avg_colors = target.mean((0, 2, 3), keepdim=True)
    loss = 0
    for i, layer in enumerate(weights):
        lam_prime = lam * ((4 ** (num_layers - i - 1)) - 1)
        layer[:, 0:3, :, :] -= avg_colors
        loss = loss + 0.5 * lam_prime * (torch.norm(layer, 2))

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


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class VGGLoss(torch.nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.model = VGG16()
        self.criterionL2 = torch.nn.MSELoss(reduction='mean')

    def forward(self, fake, target, content_weight=0.1, style_weight=1.0):
        vgg_fake = self.model(fake)
        vgg_target = self.model(target)

        content_loss = self.criterionL2(vgg_target.relu2_2, vgg_fake.relu2_2)

        # gram_matrix
        gram_style = [gram_matrix(y) for y in vgg_target]
        style_loss = 0.0
        for ft_y, gm_s in zip(vgg_fake, gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += self.criterionL2(gm_y, gm_s)

        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss
