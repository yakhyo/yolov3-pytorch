import math
from typing import Tuple

import torch

from yolov3.models.common import Concat, Conv, Detect
from torch import nn, Tensor

nc = 80  # number of classes
depth_multiple = 1.0  # model depth multiple
width_multiple = 1.0  # layer channel multiple
anchors = [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]]  # P4/16  # P5/32


class TinyBackbone(nn.Module):
    """YOLOv3 Tiny Backbone

    This backbone is built by using YOLOv3 Tiny Backbone configuration from Ultralytics.
    """

    def __init__(self):
        super().__init__()

        self.b0 = Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1)  # 0

        self.b1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 1-P1/2
        self.b2 = Conv(in_channels=16, out_channels=32, kernel_size=3, stride=1)  # 2

        self.b3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 3-P2/4
        self.b4 = Conv(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # 4

        self.b5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 5-P3/8
        self.b6 = Conv(in_channels=64, out_channels=128, kernel_size=3, stride=1)  # 6

        self.b7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 7-P4/16
        self.b8 = Conv(in_channels=128, out_channels=256, kernel_size=3, stride=1)  # 8

        self.b9 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 9-P5/32
        self.b10 = Conv(in_channels=256, out_channels=512, kernel_size=3, stride=1)  # 10
        self.b11 = nn.ZeroPad2d(padding=(0, 1, 0, 1))  # 11
        self.b12 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)  # 12

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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
        b11 = self.b11(b10)
        b12 = self.b12(b11)

        return b8, b12


class TinyHead(nn.Module):
    """YOLOv3 Tiny Head

    This TinyHead is built by using YOLOv3 Tiny Head configuration from Ultralytics.
    """

    def __init__(self):
        super().__init__()
        self.h13 = Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1)  # 13
        self.h14 = Conv(in_channels=1024, out_channels=256, kernel_size=1, stride=1)  # 14
        self.h15 = Conv(in_channels=256, out_channels=512, kernel_size=3, stride=1)  # 15 (P5/32-large)

        self.h16 = Conv(in_channels=256, out_channels=128, kernel_size=1, stride=1)  # 16
        self.h17 = nn.Upsample(None, scale_factor=2, mode="nearest")  # 17
        self.h18 = Concat(dimension=1)  # 18 cat backbone P4
        self.h19 = Conv(in_channels=384, out_channels=256, kernel_size=3, stride=1)  # 19 (P4/16-medium)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        b8, b12 = x
        h13 = self.h13(b12)
        h14 = self.h14(h13)
        h15 = self.h15(h14)

        h16 = self.h16(h14)
        h17 = self.h17(h16)
        h18 = self.h18([h17, b8])
        h19 = self.h19(h18)

        return h19, h15


class YOLOv3Tiny(nn.Module):
    def __init__(self, in_ch=3, num_classes=80, anchors=anchors):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = TinyHead()

        self.detect = Detect(anchors=anchors, nc=num_classes, ch=(256, 512))

        self.detect.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(torch.zeros(1, in_ch, 256, 256))])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self._check_anchor_order(self.detect)
        self._initialize_biases(self.detect)

    def forward(self, x):
        b8, b12 = self.backbone(x)
        h19, h15 = self.head([b8, b12])
        return self.detect([h19, h15])

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
    net = YOLOv3Tiny(anchors=anchors)
    net.eval()

    img = torch.randn(1, 3, 640, 640)
    predictions, (p3, p4) = net(img)

    print(f"P3.size(): {p3.size()}, \nP4.size(): {p4.size()}")
    print("Number of parameters: {:.2f}M".format(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6))
