"""RenderNet architecture"""
import torch
import torch.nn as nn
from base import BaseModel

class DeferredNeuralRenderer(BaseModel):
    def __init__(self, input_size):
        super(DeferredNeuralRenderer, self).__init__()
        self.stride = (2, 2)
        self.kernel = 4
        self.leakynegslope = 0.2
        self.leakyrelu = nn.LeakyReLU(self.leakynegslope)
        self.tanh = nn.Tanh()

        # TODO: Network cannot handle arbitrarily sized inputs because odd dimensions are
        #  rounded down in the convolutional layers. Need a way to fix this (asymmetric padding?)
        #  to handle arbitrary screen sizes
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=self.kernel,
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
        # TODO: Bias initialization?
        # AKA He initialization
        nn.init.kaiming_normal_(self.conv1.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv4.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5.weight, a=self.leakynegslope, nonlinearity='leaky_relu')

        # TODO: Appropriate initialization for transposed convolutions?
        nn.init.kaiming_normal_(self.conv6.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv7.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv8.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv9.weight, a=self.leakynegslope, nonlinearity='leaky_relu')

        # AKA Golrot initialization
        nn.init.xavier_uniform_(self.conv10.weight, nn.init.calculate_gain('tanh'))

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
        c6 = self.conv6(n5)
        a6 = self.leakyrelu(c6)
        n6 = self.norm6(a6)
        m7 = torch.cat((n4, n6), dim=1)
        c7 = self.conv7(m7)
        a7 = self.leakyrelu(c7)
        n7 = self.norm7(a7)
        m8 = torch.cat((n3, n7), dim=1)
        c8 = self.conv8(m8)
        a8 = self.leakyrelu(c8)
        n8 = self.norm8(a8)
        m9 = torch.cat((n2, n8), dim=1)
        c9 = self.conv9(m9)
        a9 = self.leakyrelu(c9)
        n9 = self.norm9(a9)
        m10 = torch.cat((n1, n9), dim=1)
        c10 = self.conv10(m10)
        a10 = self.tanh(c10)

        return a10
