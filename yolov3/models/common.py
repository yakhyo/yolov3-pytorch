import math
import warnings

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


def auto_pad(kernel_size: Union[int, Tuple], padding: Optional[int] = None) -> int:
    """Automatic padding to keep the size of feature map after convolution"""
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    return padding


class Conv(nn.Module):
    """Standard convolution block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
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
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard Bottleneck"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        exp: float = 0.5,
    ) -> None:
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

    def forward(self, x: torch.torch.Tensor) -> torch.torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dim = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.dim)


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer"""

    def __init__(self, in_channels: int, out_channels: int, k: Tuple[int, int, int] = (5, 9, 13)) -> None:
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
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pools], 1))


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
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


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """EMA: Exponential Moving Average"""

    def __init__(self, model, decay=0.9999, updates=0) -> None:
        # Create EMA
        self.model = deepcopy(model.module if hasattr(model, "module") else model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()  # model state_dict
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.model, model, include, exclude)


class BaseModel(nn.Module, ABC):
    """Base Model for Inheritance"""

    def __init__(self) -> None:
        super().__init__()
        if self.__class__ == BaseModel:
            warnings.warn("Don't use BaseModel directly, please use YOLOv3, YOLOv3SPP or YOLOv3Tiny instead.")

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass

    @staticmethod
    def _check_anchor_order(detect):
        # Check anchor order against stride order for  Detect() module m, and correct if necessary
        a = detect.anchors.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = detect.stride[-1] - detect.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            print("AutoAnchor: Reversing anchor order")
            detect.anchors[:] = detect.anchors.flip(0)

    @staticmethod
    def _initialize_biases(detect):
        for mi, s in zip(detect.m, detect.stride):  # from
            b = mi.bias.view(detect.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (detect.nc - 0.999999))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
