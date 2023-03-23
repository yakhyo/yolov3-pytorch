from contextlib import contextmanager

import numpy as np

import torch
from torch import distributed
from yolov3 import LOGGER
from yolov3.utils.general import colorstr


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, "module") else model.detect  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    LOGGER.info(f"\n{colorstr('AutoAnchor: ')}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). ")


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        distributed.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        distributed.barrier(device_ids=[0])
