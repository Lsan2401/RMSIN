import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from arc import AdaptiveRotatedConv2d, RountingFunction


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        routing_function1 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_4 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                                     kernel_size=3, padding=1, rounting_func=routing_function1, bias=False, kernel_number=1)

        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        routing_function2 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_3 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function2, bias=False, kernel_number=1)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        routing_function3 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_2 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function3, bias=False, kernel_number=1)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)

        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)

        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return self.conv1_1(x)
