import torch
from torch import nn
from dhg.nn import HGNNConv
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        """
               Args:
                   ``in_ch`` (``int``): :math:`C_{in}` is the number of input channels.
                   ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
                   ``n_class`` (``int``): The Number of class of the classification task.
                   ``dropout`` (``float``, optional): Dropout ratio. Defaults to 0.5.
           """
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNNConv(in_ch, n_hid)
        self.linear1=nn.Linear(n_hid, 32)

        self.linear2 = nn.Linear(32, 2)



    def forward(self, x, G):
        """The forward function.
                    Args:
                            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
                            G(``torch.Tensor``): The hypergraph structure that contains :math:`N` vertices.
                            math:  \mathbf{D}_{v}^{-\frac{1}{2}}\mathbf{HW}_e\mathbf{D}_{e}^{-1}\mathbf{H}^{\top}\mathbf{D}_{v}^{-\frac{1}{2}}
                """
        x = F.relu(self.hgc1(x, G))  # (n_node, n_hid)
        x = F.dropout(x, self.dropout)  # (n_node, n_hid)
        x_embedding = x
        x = F.relu(self.linear1(x))
        x=self.linear2(x)
        return x, x_embedding
rate = 0.0

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)
