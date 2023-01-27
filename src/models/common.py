"""Building Blocks of YOLOv3"""
from typing import List

import torch
from torch import Tensor, nn


def auto_pad(kernel_size, padding=None) -> int:
    """Remain the size of the feature map the same after convolution operation
    Args:
        kernel_size (int): kernel size of convolution
        padding (int, optional): padding size. Default: `None`
    Returns:
        padding: new padding size
    """
    if padding is None:
        padding = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else [x // 2 for x in kernel_size]
        )
    return padding


class Conv(nn.Module):
    """Standard Convolutional Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding=None,
        groups: int = 1,
        act=True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_pad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard Bottleneck Block"""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, exp=0.5):
        super().__init__()
        hidden_channels = int(out_channels * exp)  # hidden channels
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.conv2 = Conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            groups=groups,
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dim = dimension

    def forward(self, x: List[Tensor]) -> Tensor:
        return torch.cat(x, self.dim)


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) Layer"""

    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.conv2 = Conv(
            in_channels=hidden_channels * (len(k) + 1),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pools], 1))


class Detect(nn.Module):
    """Detection Head"""

    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2)
        )  # shape(nl,na,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.inplace = True

    def forward(self, x):
        """Forward"""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[
                    i
                ]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Builds a grid"""
        d = self.anchors[i].device
        x, y = torch.arange(nx).to(d), torch.arange(ny).to(d)
        yv, xv = torch.meshgrid([y, x], indexing="ij")
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid
