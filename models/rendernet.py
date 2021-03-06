"""RenderNet architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import NeuralTexture
from models import DeferredNeuralRenderer
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

