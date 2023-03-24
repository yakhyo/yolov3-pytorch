import math
from typing import List, Type

import torch
import torch.nn as nn

from yolov3.models.common import Bottleneck, Concat, Conv, Detect, SPP

# Parameters
nc = 80  # number of classes
depth_multiple = 1.0  # model depth multiple
width_multiple = 1.0  # layer channel multiple
anchors = [
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326],  # P5/32
]


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.b0 = Conv(in_channels=3, out_channels=32, kernel_size=3, stride=1)  # 0

        self.b1 = Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2)  # 1-P1/2
        self.b2 = self._make_layers(Bottleneck, channels=64, num_blocks=1)  # 2

        self.b3 = Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # 3-P2/4
        self.b4 = self._make_layers(Bottleneck, channels=128, num_blocks=2)  # 4

        self.b5 = Conv(in_channels=128, out_channels=256, kernel_size=3, stride=2)  # 5-P3/8
        self.b6 = self._make_layers(Bottleneck, channels=256, num_blocks=8)  # 6

        self.b7 = Conv(in_channels=256, out_channels=512, kernel_size=3, stride=2)  # 7-P4/14
        self.b8 = self._make_layers(Bottleneck, channels=512, num_blocks=8)  # 8

        self.b9 = Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=2)  # 9-P5/32
        self.b10 = self._make_layers(Bottleneck, channels=1024, num_blocks=4)  # 10

    @staticmethod
    def _make_layers(block: Type[Bottleneck], channels: int, num_blocks: int = 1) -> nn.Sequential:
        layers = [block(channels, channels) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b0 = self.b0(x)
        b1 = self.b1(b0)

        b2 = self.b2(b1)
        b3 = self.b3(b2)

        b4 = self.b4(b3)
        b5 = self.b5(b4)

        b6 = self.b6(b5)
        b7 = self.b7(b6)

        b8 = self.b8(b7)
        b9 = self.b9(b8)

        b10 = self.b10(b9)

        return [b6, b8, b10]


class HeadSPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h11 = Bottleneck(in_channels=1024, out_channels=1024, shortcut=False)
        self.h12 = SPP(in_channels=1024, out_channels=512, k=(5, 9, 13))
        self.h13 = Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1)
        self.h14 = Conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.h15 = Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1)  # P5/32-large

        # input of h16 is h14
        self.h16 = Conv(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.h17 = nn.Upsample(None, scale_factor=2, mode="nearest")
        self.h18 = Concat()  # cat backbone P4
        self.h19 = Bottleneck(in_channels=768, out_channels=512, shortcut=False)
        self.h20 = Bottleneck(in_channels=512, out_channels=512, shortcut=False)
        self.h21 = Conv(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.h22 = Conv(in_channels=256, out_channels=512, kernel_size=3, stride=1)  # P4/16-medium

        # inout of h23 is h21
        self.h23 = Conv(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.h24 = nn.Upsample(None, scale_factor=2, mode="nearest")
        self.h25 = Concat()  # cat backbone P3
        self.h26 = Bottleneck(in_channels=384, out_channels=256, shortcut=False)
        self.h27 = nn.Sequential(  # P3/8-small
            Bottleneck(in_channels=256, out_channels=256, shortcut=False),
            Bottleneck(in_channels=256, out_channels=256, shortcut=False),
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        b6, b8, b10 = x
        h11 = self.h11(b10)
        h12 = self.h12(h11)
        h13 = self.h13(h12)
        h14 = self.h14(h13)
        h15 = self.h15(h14)

        h16 = self.h16(h14)
        h17 = self.h17(h16)
        h18 = self.h18([h17, b8])
        h19 = self.h19(h18)
        h20 = self.h20(h19)
        h21 = self.h21(h20)
        h22 = self.h22(h21)

        h23 = self.h23(h21)
        h24 = self.h24(h23)
        h25 = self.h25([h24, b6])
        h26 = self.h26(h25)
        h27 = self.h27(h26)

        return [h27, h22, h15]


class YOLOv3SPP(nn.Module):
    def __init__(self, in_ch=3, num_classes=80, anchors=anchors):
        super().__init__()
        self.backbone = Backbone()
        self.head = HeadSPP()
        self.detect = Detect(anchors=anchors, nc=num_classes, ch=(256, 512, 1024))

        self.detect.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(torch.zeros(1, in_ch, 256, 256))])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self._check_anchor_order(self.detect)
        self._initialize_biases(self.detect)

    def forward(self, x):
        b6, b8, b10 = self.backbone(x)
        h27, h22, h15 = self.head([b6, b8, b10])
        return self.detect([h27, h22, h15])

    @staticmethod
    def _make_divisible(x, divisor):
        # Returns nearest x divisible by divisor
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

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


if __name__ == "__main__":
    net = YOLOv3SPP(anchors=anchors)
    net.eval()

    img = torch.randn(1, 3, 640, 640)
    predictions, (p3, p4, p5) = net(img)

    print(f"P3.size(): {p3.size()}, \nP4.size(): {p4.size()}, \nP5.size(): {p5.size()}")
    print("Number of parameters: {:.2f}M".format(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6))
