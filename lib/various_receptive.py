import torch
from torch import mean, nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from numpy import random
import os



def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def _conv_bn(input_channel,output_channel,kernel_size=3,padding=1,stride=1,groups=1):
     res=nn.Sequential()
     res.add_module('conv',nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size,padding=padding,padding_mode='zeros',stride=stride,groups=groups,bias=False))
     res.add_module('bn',nn.BatchNorm2d(output_channel))
     return res


def _conv_bn2(input_channel,output_channel,kernel_size=3,padding=1,stride=1,groups=1):
     res=nn.Sequential()
     res.add_module('conv1',nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,padding_mode='zeros',stride=stride,groups=groups,bias=False))
     res.add_module('bn1',nn.BatchNorm2d(output_channel))
     res.add_module('conv2',nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size,padding=padding,padding_mode='zeros',stride=stride,groups=groups,bias=False))
     res.add_module('bn2',nn.BatchNorm2d(output_channel))
     return res


class RepBlock(nn.Module):
     def __init__(self,input_channel,output_channel,kernel_size=3,groups=1,stride=1):
          super().__init__()
          self.input_channel=input_channel
          self.output_channel=output_channel
          self.kernel_size=kernel_size
          self.padding=kernel_size//2
          self.groups=groups
          self.activation=nn.ReLU()
          self.sigmoid=nn.Sigmoid()

          #make sure kernel_size=3 padding=1
          assert self.kernel_size==3
          assert self.padding==1

          self.brb_3x3=_conv_bn2(input_channel,output_channel,kernel_size=self.kernel_size,padding=self.padding,groups=groups)
          self.brb_1x1=_conv_bn(input_channel,output_channel,kernel_size=1,padding=0,groups=groups)
          self.brb_identity=nn.BatchNorm2d(self.input_channel) if self.input_channel == self.output_channel else None

          self.brb_3x3_2=_conv_bn2(input_channel,output_channel,kernel_size=self.kernel_size,padding=self.padding,groups=groups)
          self.brb_1x1_2=_conv_bn(input_channel,output_channel,kernel_size=1,padding=0,groups=groups)
          self.brb_identity_2=nn.BatchNorm2d(self.input_channel) if self.input_channel == self.output_channel else None

     def forward(self, inputs):
          if(self.brb_identity==None):
               identity_out=0
          else:
               identity_out=self.brb_identity(inputs)
          out1=self.activation(self.brb_1x1(inputs)+self.brb_3x3(inputs)+identity_out)


          if(self.brb_identity_2==None):
               identity_out_2=0
          else:
               identity_out_2=self.brb_identity_2(out1)
          out2=self.brb_1x1_2(out1)+self.brb_3x3_2(out1)+identity_out_2

          # print('relu')

          return self.sigmoid(out2)


class VariousReceptive(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.repblock = RepBlock(1, 1)
    
    def forward(self, x):
        bs, n, dim = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))

        input = x.view(bs, h, w, dim).permute(0, 3, 1, 2)  # bs,dim,h,w
        mean_input = torch.mean(input,dim=1,keepdim=True)  # bs,1,h,w
        weight = self.repblock(mean_input)  # bs,1,h,w
        out = input * weight
        out = out.reshape(bs, dim, -1).permute(0, 2, 1)   # bs,n,dim
        return out


###test
if __name__ == '__main__':
    input=torch.randn(50,1,49,49)
    repblock=RepBlock(1,1)
    repblock.eval()
    out=repblock(input)

    