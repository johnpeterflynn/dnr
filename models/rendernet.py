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

        #t1 = torch.cuda.Event(enable_timing=True)
        #t2 = torch.cuda.Event(enable_timing=True)
        #t1.record()
        f = self.dnr(f)
        #t2.record()

        #torch.cuda.synchronize()
        #print('dnr:', t1.elapsed_time(t2))

        return f


class NeuralTexture(nn.Module):
    def __init__(self, size, depth, mipmap_levels):
        super(NeuralTexture, self).__init__()
        self.depth = depth

        # Init between [-1,1] to match our output texture
        # dim0 = 1 to be dimensionally compatible with grid_sample
        #self.mipmap = nn.ParameterList([nn.Parameter(
        #    torch.FloatTensor(1, depth, int(size / (2 ** i)), int(size / (2 ** i))).uniform_(-1, 1),
        #    requires_grad=True) for i in range(mipmap_levels)])

        # NOTE: TorchScript doesn't currently support nn.ParameterList so we need to unroll it for now
        self.mipmap_0 = nn.Parameter(
            torch.FloatTensor(1, depth, int(size / (2 ** 0)), int(size / (2 ** 0))).uniform_(-1, 1),
            requires_grad=True)
        self.mipmap_1 = nn.Parameter(
            torch.FloatTensor(1, depth, int(size / (2 ** 1)), int(size / (2 ** 1))).uniform_(-1, 1),
            requires_grad=True)
        self.mipmap_2 = nn.Parameter(
            torch.FloatTensor(1, depth, int(size / (2 ** 2)), int(size / (2 ** 2))).uniform_(-1, 1),
            requires_grad=True)
        self.mipmap_3 = nn.Parameter(
            torch.FloatTensor(1, depth, int(size / (2 ** 3)), int(size / (2 ** 3))).uniform_(-1, 1),
            requires_grad=True)

    def forward(self, input):
        # TODO: NOTE: grid_sample samples texture as (row, col) while uv coordinates are
        #  given as (u, v) s.t. (row, col) == (2*v - 1, 2*u - 1). We'll convert the range
        #  from [0,1] to [-1,1] like above but we won't bother swapping u and v (for now).
        #  This shouldn't be a problem as long as we're consistent but should eventually
        #  be fixed for correctness.
        # TODO: Is training slowed by averaging over zeros that exist due to some pixels that
        #  have yet tgo be trained?
        # Convert from [0, 1] to [-1, 1]
        grid = 2 * input - 1
        n_batches, _, _, _ = grid.shape
        sample = 0
        #for texture in self.mipmap:
        #    sample += F.grid_sample(texture.expand(n_batches, -1, -1, -1), grid, align_corners=False)

        #t1 = torch.cuda.Event(enable_timing=True)
        #t1.record()
        sample += F.grid_sample(self.mipmap_0.expand(n_batches, -1, -1, -1), grid, align_corners=False)
        #t2 = torch.cuda.Event(enable_timing=True)
        #t2.record()
        sample += F.grid_sample(self.mipmap_1.expand(n_batches, -1, -1, -1), grid, align_corners=False)
        #t3 = torch.cuda.Event(enable_timing=True)
        #t3.record()
        sample += F.grid_sample(self.mipmap_2.expand(n_batches, -1, -1, -1), grid, align_corners=False)
        #t4 = torch.cuda.Event(enable_timing=True)
        #t4.record()
        sample += F.grid_sample(self.mipmap_3.expand(n_batches, -1, -1, -1), grid, align_corners=False)
        #t5 = torch.cuda.Event(enable_timing=True)
        #t5.record()

        #torch.cuda.synchronize()
        #print('Sample 1', t1.elapsed_time(t2))
        #print('Sample 2', t2.elapsed_time(t3))
        #print('Sample 3', t3.elapsed_time(t4))
        #print('Sample 4', t4.elapsed_time(t5))
        #print('Sanity', t1.elapsed_time(t5))

        return sample

    def get_mipmap(self):
        return [self.mipmap_0, self.mipmap_1, self.mipmap_2, self.mipmap_3]


class DeferredNeuralRenderer(nn.Module):
    def __init__(self, input_channels, output_channels=3):
        super(DeferredNeuralRenderer, self).__init__()
        base_channels = 64
        self.stride = (2, 2)
        self.kernel = 4
        self.leakynegslope = 0.2
        self.leakyrelu = nn.LeakyReLU(self.leakynegslope, True)
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm1 = nn.InstanceNorm2d(num_features=base_channels)
        self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm2 = nn.InstanceNorm2d(num_features=base_channels * 2)
        self.conv3 = nn.Conv2d(in_channels=base_channels * 2, out_channels=base_channels * 4, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm3 = nn.InstanceNorm2d(num_features=base_channels * 4)
        self.conv4 = nn.Conv2d(in_channels=base_channels * 4, out_channels=base_channels * 8, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm4 = nn.InstanceNorm2d(num_features=base_channels * 8)
        self.conv5 = nn.Conv2d(in_channels=base_channels * 8, out_channels=base_channels * 8, kernel_size=self.kernel,
                               stride=self.stride, padding=1)
        self.norm5 = nn.InstanceNorm2d(num_features=base_channels * 8)

        self.conv6 = nn.ConvTranspose2d(in_channels=base_channels * 8, out_channels=base_channels * 8, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm6 = nn.InstanceNorm2d(num_features=base_channels * 8)
        self.conv7 = nn.ConvTranspose2d(in_channels=base_channels * 8 * 2, out_channels=base_channels * 4, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm7 = nn.InstanceNorm2d(num_features=base_channels * 4)
        self.conv8 = nn.ConvTranspose2d(in_channels=base_channels * 4 * 2, out_channels=base_channels * 2, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm8 = nn.InstanceNorm2d(num_features=base_channels * 2)
        self.conv9 = nn.ConvTranspose2d(in_channels=base_channels * 2 * 2, out_channels=base_channels, kernel_size=self.kernel,
                                        stride=self.stride, padding=1)
        self.norm9 = nn.InstanceNorm2d(num_features=base_channels)
        self.conv10 = nn.ConvTranspose2d(in_channels=base_channels * 2, out_channels=output_channels, kernel_size=self.kernel,
                                         stride=self.stride, padding=1)

        # TODO: _normal_ vs _uniform_? 'fan_in' vs 'fan_out'?
        # AKA He initialization
        nn.init.kaiming_normal_(self.conv1.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv4.weight, a=self.leakynegslope, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')

        # TODO: Is this a good bias initialization for relu / leaky relu?
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.conv5.bias)

        # TODO: Appropriate initialization for transposed convolutions?
        nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv8.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv9.weight, nonlinearity='leaky_relu')

        # TODO: Is this a good bias initialization for relu?
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
        #t1 = torch.cuda.Event(enable_timing=True)
        #t1.record()

        c1 = self.conv1(input)
        n1 = self.norm1(c1)
        a1 = self.leakyrelu(n1)

        #t2 = torch.cuda.Event(enable_timing=True)
        #t2.record()

        c2 = self.conv2(a1)
        n2 = self.norm2(c2)
        a2 = self.leakyrelu(n2)

        #t3 = torch.cuda.Event(enable_timing=True)
        #t3.record()

        c3 = self.conv3(a2)
        n3 = self.norm3(c3)
        a3 = self.leakyrelu(n3)

        #t4 = torch.cuda.Event(enable_timing=True)
        #t4.record()

        c4 = self.conv4(a3)
        n4 = self.norm4(c4)
        a4 = self.leakyrelu(n4)

        #t5 = torch.cuda.Event(enable_timing=True)
        #t5.record()

        c5 = self.conv5(a4)
        n5 = self.norm5(c5)
        a5 = self.leakyrelu(n5)

        #t6 = torch.cuda.Event(enable_timing=True)
        #t6.record()

        # TODO: Verify that this is the correct order for skip connections
        c6 = self.conv6(a5, output_size=c4.shape)
        
        #t6_2 = torch.cuda.Event(enable_timing=True)
        #t6_2.record()
        
        n6 = self.norm6(c6)
        a6 = self.leakyrelu(n6)
         
        #t7 = torch.cuda.Event(enable_timing=True)
        #t7.record()

        m7 = torch.cat((a4, a6), dim=1)
        
        #t7_1 = torch.cuda.Event(enable_timing=True)
        #t7_1.record()

        c7 = self.conv7(m7, output_size=c3.shape)

        #t7_2 = torch.cuda.Event(enable_timing=True)
        #t7_2.record()

        n7 = self.norm7(c7)        
        a7 = self.leakyrelu(n7)

        #t8 = torch.cuda.Event(enable_timing=True)
        #t8.record()

        m8 = torch.cat((a3, a7), dim=1)
        
        #t8_1 = torch.cuda.Event(enable_timing=True)
        #t8_1.record()

        c8 = self.conv8(m8, output_size=c2.shape)
                
        #t8_2 = torch.cuda.Event(enable_timing=True)
        #t8_2.record()

        n8 = self.norm8(c8) 
        a8 = self.leakyrelu(n8)

        #t9 = torch.cuda.Event(enable_timing=True)
        #t9.record()

        m9 = torch.cat((a2, a8), dim=1)
        
        #t9_1 = torch.cuda.Event(enable_timing=True)
        #t9_1.record()

        c9 = self.conv9(m9, output_size=c1.shape)
        
        #t9_11 = torch.cuda.Event(enable_timing=True)
        #t9_11.record()
        
        n9 = self.norm9(c9)
        
        #t9_2 = torch.cuda.Event(enable_timing=True)
        #t9_2.record()
        
        a9 = self.leakyrelu(n9)
        
        #t10 = torch.cuda.Event(enable_timing=True)
        #t10.record()

        m10 = torch.cat((a1, a9), dim=1)
        
        #t10_1 = torch.cuda.Event(enable_timing=True)
        #t10_1.record()

        c10 = self.conv10(m10, output_size=input.shape)
        a10 = self.tanh(c10)

        #t11 = torch.cuda.Event(enable_timing=True)
        #t11.record()

        #torch.cuda.synchronize()
        #print('t2', t1.elapsed_time(t2))
        #print('t3', t2.elapsed_time(t3))
        #print('t4', t3.elapsed_time(t4))
        #print('t5', t4.elapsed_time(t5))
        #print('t6', t5.elapsed_time(t6))
        #print('t6_2', t6.elapsed_time(t6_2))
        #print('t7', t6_2.elapsed_time(t7))
        #print('t7_1', t7.elapsed_time(t7_1))
        #print('t7_2', t7_1.elapsed_time(t7_2))
        #print('t8', t7_2.elapsed_time(t8))
        #print('t8_1', t8.elapsed_time(t8_1))
        #print('t8_2', t8_1.elapsed_time(t8_2))
        #print('t9', t8_2.elapsed_time(t9))
        #print('t9_1', t9.elapsed_time(t9_1))
        #print('t9_11', t9_1.elapsed_time(t9_11))
        #print('t9_2', t9_11.elapsed_time(t9_2))
        #print('t10', t9_2.elapsed_time(t10))
        #print('t10_1', t10.elapsed_time(t10_1))
        #print('t11', t10_1.elapsed_time(t11))

        return a10
