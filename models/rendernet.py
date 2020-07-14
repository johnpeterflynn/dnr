"""RenderNet architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class RenderNet(BaseModel):
    def __init__(self, texture_size, texture_depth, mipmap_levels):
        super(RenderNet, self).__init__()
        self.neural_texture = NeuralTexture(texture_size, texture_depth, mipmap_levels)
        self.dnr = DeferredNeuralRenderer(texture_depth)

    def forward(self, input):
        f = self.neural_texture(input)
        f = self.dnr(f)

        return f


class NeuralTexture(nn.Module):
    def __init__(self, size, depth, mipmap_levels):
        super(NeuralTexture, self).__init__()
        # Init between [-1,1] to match our output texture
        # dim0 = 1 to be dimensionally compatible with grid_sample
        self.depth = depth

        self.mipmap = nn.ParameterList([nn.Parameter(
            torch.FloatTensor(1, depth, int(size / (2 ** i)), int(size / (2 ** i))).uniform_(-1, 1),
            requires_grad=True) for i in range(mipmap_levels)])

    def forward(self, input):
        # TODO: Make self.textures work with batch_size > 1
        # TODO: NOTE: grid_sample samples texture as (row, col) while uv coordinates are
        #  given as (u, v) s.t. (row, col) == (2*v - 1, 2*u - 1). We'll convert the range
        #  from [0,1] to [-1,1] like above but we won't bother swapping u and v (for now).
        #  This shouldn't be a problem as long as we're consistent but should eventually
        #  be fixed for correctness.
        #grid = 2 * input - 1
        #sample = F.grid_sample(self.texture, grid, align_corners=False)

        #grid = 2 * input - 1
        #for i in range(grid.shape[0]):
        #    s = F.grid_sample(self.texture, grid[i, :, :, :].unsqueeze(0), align_corners=False)
        #
        #    if i == 0:
        #        samples = s
        #    else:
        #        samples = torch.cat((samples, s), dim=0)
        #
        #return samples

        # TODO: Is training slowed by averaging over zeros that exist due to some pixels that
        #  have yet tgo be trained?
        # Convert from [0, 1] to [-1, 1]
        grid = 2 * input - 1
        n_batches, _, _, _ = grid.shape
        sample = 0
        for texture in self.mipmap:
            sample += F.grid_sample(texture.expand(n_batches, -1, -1, -1), grid, align_corners=False)

        return sample


class DeferredNeuralRenderer(nn.Module):
    def __init__(self, input_channels):
        super(DeferredNeuralRenderer, self).__init__()
        self.stride = (2, 2)
        self.kernel = 4
        self.leakynegslope = 0.2
        self.leakyrelu = nn.LeakyReLU(self.leakynegslope)
        self.tanh = nn.Tanh()

        # NOTE: Arbitrary screen side made possible by dynamic output padding.
        # TODO: Output padding should be an input parameter

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm1 = nn.InstanceNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm2 = nn.InstanceNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm3 = nn.InstanceNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm4 = nn.InstanceNorm2d(num_features=512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm5 = nn.InstanceNorm2d(num_features=512)

        self.conv6 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm6 = nn.InstanceNorm2d(num_features=512)
        self.conv7 = nn.ConvTranspose2d(in_channels=512+512, out_channels=512, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm7 = nn.InstanceNorm2d(num_features=512)
        self.conv8 = nn.ConvTranspose2d(in_channels=512+256, out_channels=256, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm8 = nn.InstanceNorm2d(num_features=256)
        self.conv9 = nn.ConvTranspose2d(in_channels=256+128, out_channels=128, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm9 = nn.InstanceNorm2d(num_features=128)
        self.conv10 = nn.ConvTranspose2d(in_channels=128+64, out_channels=3, kernel_size=self.kernel,
                                         stride=self.stride, padding=1)

        # TODO: _normal_ vs _uniform_? 'fan_in' vs 'fan_out'?
        # AKA He initialization
        nn.init.kaiming_normal_(self.conv1.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv4.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5.weight, a=self.leakynegslope, nonlinearity='leaky_relu')

        # TODO: Is this a good bias initialization for leaky relu?
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.conv5.bias)

        # TODO: Appropriate initialization for transposed convolutions?
        nn.init.kaiming_normal_(self.conv6.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv7.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv8.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv9.weight, a=self.leakynegslope, nonlinearity='leaky_relu')

        # TODO: Is this a good bias initialization for leaky relu?
        nn.init.zeros_(self.conv6.bias)
        nn.init.zeros_(self.conv7.bias)
        nn.init.zeros_(self.conv8.bias)
        nn.init.zeros_(self.conv9.bias)

        # AKA Golrot initialization
        nn.init.xavier_uniform_(self.conv10.weight, nn.init.calculate_gain('tanh'))

        # Set initial bias of output layer to zero since we're using (anti)symmetric tanh activation
        nn.init.zeros_(self.conv10.bias)

    def forward(self, input):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """
        c1 = self.conv1(input)
        a1 = self.leakyrelu(c1)
        n1 = self.norm1(a1)
        c2 = self.conv2(n1)
        a2 = self.leakyrelu(c2)
        n2 = self.norm2(a2)
        c3 = self.conv3(n2)
        a3 = self.leakyrelu(c3)
        n3 = self.norm3(a3)
        c4 = self.conv4(n3)
        a4 = self.leakyrelu(c4)
        n4 = self.norm4(a4)
        c5 = self.conv5(n4)
        a5 = self.leakyrelu(c5)
        n5 = self.norm5(a5)

        # TODO: Verify that this is the correct order for skip connections
        c6 = self.conv6(n5, output_size=c4.shape)
        a6 = self.leakyrelu(c6)
        n6 = self.norm6(a6)
        m7 = torch.cat((n4, n6), dim=1)
        c7 = self.conv7(m7, output_size=c3.shape)
        a7 = self.leakyrelu(c7)
        n7 = self.norm7(a7)
        m8 = torch.cat((n3, n7), dim=1)
        c8 = self.conv8(m8, output_size=c2.shape)
        a8 = self.leakyrelu(c8)
        n8 = self.norm8(a8)
        m9 = torch.cat((n2, n8), dim=1)
        c9 = self.conv9(m9, output_size=c1.shape)
        a9 = self.leakyrelu(c9)
        n9 = self.norm9(a9)
        m10 = torch.cat((n1, n9), dim=1)
        c10 = self.conv10(m10, output_size=input.shape)
        a10 = self.tanh(c10)

        return a10
