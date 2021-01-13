"""RenderNet architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import NeuralTexture
from models import DeferredNeuralRenderer
from base import BaseModel


class RenderNetNoUNet(BaseModel):
    def __init__(self, texture_size, texture_depth, mipmap_levels):
        super(RenderNetNoUNet, self).__init__()
        self.neural_texture = NeuralTexture(texture_size, texture_depth, mipmap_levels)
        #self.dnr = DeferredNeuralRenderer(texture_depth)
        self.dummy = DummyModule()

    def forward(self, input):
        f = self.neural_texture(input)

        #t1 = torch.cuda.Event(enable_timing=True)
        #t2 = torch.cuda.Event(enable_timing=True)
        #t1.record()
        #f = self.dnr(f)
        #t2.record()

        #torch.cuda.synchronize()
        #print('dnr:', t1.elapsed_time(t2))

        return f

# TODO: WARNING: Another temporary hack to get around the strict creation of an
#  optimizer at program init
class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
        self.dummy = nn.Parameter(torch.rand(10, requires_grad=True))

