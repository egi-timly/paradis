import torch
import torch.nn as nn
from torch.autograd import Variable

from ..core.primitives import AbstractCombOp, AbstractPrimitive

def darts_concat(tensors):

    return(torch.cat([s for s in tensors], dim=1))


class DARTSConcat(AbstractCombOp):

    def __init__(self):
        super().__init__(comb_op=darts_concat)

    def __call__(self, tensors, edges_data=None):
        out = self.comb_op(tensors)
        return(out)
    

class FactorizedReduce(AbstractPrimitive):

    def __init__(self, C_in, C_out, affine=True, **kwargs):
        super().__init__(locals())
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, edge_data=None):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out
  
    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None
    
class AuxiliaryHeadCIFAR(AbstractPrimitive):

    def __init__(self, C, num_classes, **kwargs):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x, edge_data=None):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None