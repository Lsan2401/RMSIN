import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class Residual(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class ScaleAwareGate(nn.Module):
    def __init__(self, inp, oup):
        super(ScaleAwareGate, self).__init__()

        self.local_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(oup)

        self.global_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(oup)

        self.global_act = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(oup)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        local_feat = self.bn1(local_feat)

        global_feat = self.global_embedding(x_g)
        global_feat = self.bn2(global_feat)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        global_act = self.global_act(x_g)
        global_act = self.bn3(global_act)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class Attention(torch.nn.Module):
    def __init__(self, dim, img_shape, att_shape, key_dim=32, num_heads=8, attn_ratio=2, activation=torch.nn.Hardswish):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.img_shape = img_shape
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h)

        self.parallel_conv = nn.Sequential(
            nn.Hardswish(inplace=False),
            nn.Conv2d(self.dh, self.dh, kernel_size=3, padding=1, groups=self.dh),
        )
        self.to_out = nn.Linear(self.dh, dim)
        self.proj = nn.Linear(att_shape, img_shape)

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        v0 = v[:, :, :self.img_shape, :]

        v0 = v0.reshape(B, self.dh, int(self.img_shape ** 0.5), -1)
        v_conv = self.parallel_conv(v0).flatten(2)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, N)
        x = self.proj(x) + v_conv
        x = self.to_out(x.permute(0, 2, 1))  # + v_conv
        return x


class CrossScaleAttention(nn.Module):
    def __init__(self, dim, img_shape=225, att_shape=314):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)

        self.DWConv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
        )
        self.DWConv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=3, padding=2, groups=dim),
            nn.BatchNorm2d(dim),
        )
        self.attention = Attention(dim, img_shape, att_shape)
        self.bn4 = nn.BatchNorm2d(dim)
        self.activate = nn.Hardswish()
        self.conv = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        x0 = self.bn1(x)
        x1 = self.DWConv1(x0)
        x2 = self.DWConv2(x0)
        # [B, C, H, W] -> [B, C, H*W]
        x0, x1, x2 = x0.view(x0.shape[0], x0.shape[1], -1), x1.view(x1.shape[0], x1.shape[1], -1), x2.view(x2.shape[0], x2.shape[1], -1)
        attn = torch.cat((x0, x1, x2), dim=2).permute(0, 2, 1)
        attn = self.attention(attn)
        attn = attn.permute(0, 2, 1).contiguous().view(x0.shape[0], x0.shape[1], 15, 15)
        x = self.conv(self.activate(self.bn4(attn)))
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=dim)
        self.relu = nn.ReLU6()
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        out = self.conv3(self.relu(self.conv2(self.bn2(self.conv1(self.bn1(x))))))
        return out


class IntraFeedForward(nn.Module):
    def __init__(self, channels, mlp_ratio=2):
        super().__init__()
        self.channels = [channels[i]//4 for i in range(len(channels))]

        self.ff1 = Residual(FeedForward(self.channels[0], mlp_ratio*self.channels[0]))
        self.ff2 = Residual(FeedForward(self.channels[1], mlp_ratio*self.channels[1]))
        self.ff3 = Residual(FeedForward(self.channels[2], mlp_ratio*self.channels[2]))
        self.ff4 = Residual(FeedForward(self.channels[3], mlp_ratio*self.channels[3]))

    def forward(self, x):
        x1, x2, x3, x4 = x.split(self.channels, dim=1)
        x1 = self.ff1(x1)
        x2 = self.ff2(x2)
        x3 = self.ff3(x3)
        x4 = self.ff4(x4)
        return torch.cat([x1, x2, x3, x4], dim=1)


class CIMBlock(nn.Module):
    def __init__(self, dim, channels, mlp_ratio=2):
        super().__init__()
        self.csa1 = Residual(CrossScaleAttention(dim))
        self.intra_ff = Residual(IntraFeedForward(channels, mlp_ratio))
        self.csa2 = Residual(CrossScaleAttention(dim))
        self.ff = Residual(FeedForward(dim, dim*mlp_ratio))

    def forward(self, x):
        x = self.csa1(x)
        x = self.intra_ff(x)
        x = self.csa2(x)
        x = self.ff(x)
        return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class CIM(nn.Module):
    def __init__(self, dim, num_layers=1, channels=[128, 256, 512, 1024], downsample=1):
        super().__init__()
        self.hidden_dim = dim // 4
        self.channels = channels
        self.stride = downsample

        self.down_channel = nn.Conv2d(dim, self.hidden_dim, 1)
        self.up_channel = nn.Conv2d(self.hidden_dim, dim, 1)

        # downsample to h/32, w/32
        self.pool = PyramidPoolAgg(stride=self.stride)
        self.block = nn.ModuleList([
            CIMBlock(self.hidden_dim, channels)
            for _ in range(num_layers)
        ])
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.fusion = nn.ModuleList([
            ScaleAwareGate(channels[i], channels[i])  
            for i in range(len(channels))
        ])

    def forward(self, input):  # [B, C, H, W]
        out = self.pool(input)
        out = self.down_channel(out)
        for layer in self.block:
            out = layer(out)
        out = self.bn(out)
        out = self.up_channel(out)
        xx =  out.split(self.channels, dim=1)
        results = []
        for i in range(len(self.channels)):
            CIM_before = input[i]
            CIM_after = xx[i]
            out_ = self.fusion[i](CIM_before, CIM_after)
            results.append(out_)
        return results



if __name__ == '__main__':
    model = CIM(1920)
    x1 = torch.randn(2, 128, 120, 120)
    x2 = torch.randn(2, 256, 60, 60)
    x3 = torch.randn(2, 512, 30, 30)
    x4 = torch.randn(2, 1024, 15, 15)
    x = tuple([x1, x2, x3, x4])
    y = model(x)


