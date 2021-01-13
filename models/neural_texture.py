import torch
import torch.nn as nn
import torch.nn.functional as F


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



