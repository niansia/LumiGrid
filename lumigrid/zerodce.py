import torch
import torch.nn as nn


class ZeroDCE(nn.Module):
    """DCE-Net from Guo et al., 'Zero-Reference Deep Curve Estimation' (CVPR 2020): 7 convs predicting 8 pixel-wise
    quadratic light-enhancement curves LE(x) = x + a * x * (1 - x), applied iteratively."""

    def __init__(self, nf: int = 32):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x)); x2 = self.relu(self.e_conv2(x1)); x3 = self.relu(self.e_conv3(x2)); x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1))); x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for a in torch.chunk(r, 8, 1):
            x = x + a * (x * x - x)
        return x.clamp(0, 1)
