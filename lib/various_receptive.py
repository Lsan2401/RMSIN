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




def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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
     def __init__(self,input_channel,output_channel,kernel_size=3,groups=1,stride=1,deploy=False):
          super().__init__()
          self.input_channel=input_channel
          self.output_channel=output_channel
          self.deploy=deploy
          self.kernel_size=kernel_size
          self.padding=kernel_size//2
          self.groups=groups
          self.activation=nn.ReLU()
          self.sigmoid=nn.Sigmoid()

          #make sure kernel_size=3 padding=1
          assert self.kernel_size==3
          assert self.padding==1
          if(not self.deploy):
               self.brb_3x3=_conv_bn2(input_channel,output_channel,kernel_size=self.kernel_size,padding=self.padding,groups=groups)
               self.brb_1x1=_conv_bn(input_channel,output_channel,kernel_size=1,padding=0,groups=groups)
               self.brb_identity=nn.BatchNorm2d(self.input_channel) if self.input_channel == self.output_channel else None
          else:
               self.brb_rep=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=self.kernel_size,padding=self.padding,padding_mode='zeros',stride=stride,bias=True)

          if(not self.deploy):
               self.brb_3x3_2=_conv_bn2(input_channel,output_channel,kernel_size=self.kernel_size,padding=self.padding,groups=groups)
               self.brb_1x1_2=_conv_bn(input_channel,output_channel,kernel_size=1,padding=0,groups=groups)
               self.brb_identity_2=nn.BatchNorm2d(self.input_channel) if self.input_channel == self.output_channel else None
          else:
               self.brb_rep_2=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=self.kernel_size,padding=self.padding,padding_mode='zeros',stride=stride,bias=True)


     
     def forward(self, inputs):
          if(self.deploy):
               return self.sigmoid(self.brb_rep_2(self.activation(self.brb_rep(inputs))))
          
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

     def _switch_to_deploy(self):
          self.deploy=True
          kernel,bias=self._get_equivalent_kernel_bias()
          kernel_2,bias_2=self._get_equivalent_kernel_bias_2()
          self.brb_rep=nn.Conv2d(in_channels=self.brb_3x3.conv2.in_channels,out_channels=self.brb_3x3.conv2.out_channels,
                                   kernel_size=self.brb_3x3.conv2.kernel_size,padding=self.brb_3x3.conv2.padding,
                                   padding_mode=self.brb_3x3.conv2.padding_mode,stride=self.brb_3x3.conv2.stride,
                                   groups=self.brb_3x3.conv2.groups,bias=True)

          self.brb_rep_2=nn.Conv2d(in_channels=self.brb_3x3.conv2.in_channels,out_channels=self.brb_3x3.conv2.out_channels,
                                   kernel_size=self.brb_3x3.conv2.kernel_size,padding=self.brb_3x3.conv2.padding,
                                   padding_mode=self.brb_3x3.conv2.padding_mode,stride=self.brb_3x3.conv2.stride,
                                   groups=self.brb_3x3.conv2.groups,bias=True)


          self.brb_rep.weight.data=kernel
          self.brb_rep.bias.data=bias
          self.brb_rep_2.weight.data=kernel_2
          self.brb_rep_2.bias.data=bias_2

          #Eliminate gradient update
          for para in self.parameters():
               para.detach_()
          #Delete useless branches
          self.__delattr__('brb_3x3')
          self.__delattr__('brb_1x1')
          self.__delattr__('brb_identity')
          self.__delattr__('brb_3x3_2')
          self.__delattr__('brb_1x1_2')
          self.__delattr__('brb_identity_2')

     #Turn 1x1 convolution into 3x3 convolution
     def _pad_1x1_kernel(self,kernel):
          if(kernel is None):
               return 0
          else:
               return F.pad(kernel,[1]*4)


     #The convolutions of identity, 1x1 and 3x3 are fused together to become a parameter of 3x3 convolution
     def _get_equivalent_kernel_bias(self):
          brb_3x3_weight,brb_3x3_bias=self._fuse_conv_bn(self.brb_3x3)
          brb_1x1_weight,brb_1x1_bias=self._fuse_conv_bn(self.brb_1x1)
          brb_id_weight,brb_id_bias=self._fuse_conv_bn(self.brb_identity)
          return brb_3x3_weight+self._pad_1x1_kernel(brb_1x1_weight)+brb_id_weight,brb_3x3_bias+brb_1x1_bias+brb_id_bias

     def _get_equivalent_kernel_bias_2(self):
          brb_3x3_weight,brb_3x3_bias=self._fuse_conv_bn(self.brb_3x3_2)
          brb_1x1_weight,brb_1x1_bias=self._fuse_conv_bn(self.brb_1x1_2)
          brb_id_weight,brb_id_bias=self._fuse_conv_bn(self.brb_identity_2)
          return brb_3x3_weight+self._pad_1x1_kernel(brb_1x1_weight)+brb_id_weight,brb_3x3_bias+brb_1x1_bias+brb_id_bias
          
     
     ### The parameters of convolution and BN are fused together
     def _fuse_conv_bn(self,branch):
          if(branch is None):
               return 0,0
          elif(isinstance(branch,nn.Sequential)):
               if(len(branch)==2):
                    kernel=branch.conv.weight
                    running_mean=branch.bn.running_mean
                    running_var=branch.bn.running_var
                    gamma=branch.bn.weight
                    beta=branch.bn.bias
                    eps=branch.bn.eps
               else:
                    fisrt_kernel, fisrt_bias = transI_fusebn(branch.conv1.weight, branch.bn1)
                    second_kernel, second_bias = transI_fusebn(branch.conv2.weight, branch.bn2)
                    kernel,bias= transIII_1x1_kxk(fisrt_kernel, fisrt_bias, second_kernel, second_bias, groups=self.groups)
                    return kernel,bias
          else:
               assert isinstance(branch, nn.BatchNorm2d)
               if not hasattr(self, 'id_tensor'):
                    input_dim = self.input_channel // self.groups
                    kernel_value = np.zeros((self.input_channel, input_dim, 3, 3), dtype=np.float32)
                    for i in range(self.input_channel):
                         kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
               kernel = self.id_tensor
               running_mean = branch.running_mean
               running_var = branch.running_var
               gamma = branch.weight
               beta = branch.bias
               eps = branch.eps
          
          std=(running_var+eps).sqrt()
          t=gamma/std
          t=t.view(-1,1,1,1)
          return kernel*t,beta-running_mean*gamma/std
          

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
    repblock._switch_to_deploy()
    out2=repblock(input)

    