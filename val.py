import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from yolov3.models import YOLOv3SPP

from yolov3 import LOGGER
from yolov3.utils.datasets import create_dataloader
from yolov3.utils.general import check_img_size, colorstr, non_max_suppression, scale_boxes, xywh2xyxy
from yolov3.utils.metrics import ap_per_class, box_iou


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections: (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels: (Array[M, 5]), class, x1, y1, x2, y2
        iouv: iou vector for mAP@0.5:0.95
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    half=True,  # use FP16 half-precision inference
    model=None,
    dataloader=None,
    compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device, PyTorch model

        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model

        model = YOLOv3SPP().to(device)
        checkpoint = torch.load(weights, map_location=device)
        names = checkpoint["model"].names
        state_dict = checkpoint["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
        model.names = names

        stride = int(model.detect.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        half = True
        model.half() if half else model.float()

        # Data
        import yaml

        if isinstance(data, (str, Path)):
            with open(data, errors="ignore") as f:
                data = yaml.safe_load(f)  # dictionary

        # Parse yaml
        path = Path(data.get("path"))  # optional 'path' default to '.'
        for k in "train", "val", "test":
            if data.get(k):  # prepend path
                data[k] = str(path / data[k])

    # Configure
    model.eval()
    nc = int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # warmup
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader, _ = create_dataloader(data[task], imgsz, batch_size, stride, prefix=colorstr(f"{task}: "))

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, "names") else model.module.names)}

    desc = ("%20s" + "%11s" * 6) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95")
    dt, p, r, f1, mp, mr, map50, map = ([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    for batch_i, (images, targets, paths, shapes) in enumerate(pbar):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device)

        images = images.half() if half else images.float()  # uint8 to fp16/32
        images /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = images.shape  # batch size, channels, height, width

        # Inference
        outputs, train_out = model(images) if training else model(images)  # inference, loss outputs

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        outputs = non_max_suppression(outputs, conf_thres, iou_thres, multi_label=True)

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            path, shape = Path(paths[i]), shapes[i][0]
            correct = torch.zeros(output.shape[0], niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if output.shape[0] == 0:  # number of predictions
                if labels.shape[0]:  # number of labels
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels[:, 0]))
                continue

            # Predictions
            predictions = output.clone()
            scale_boxes(images[i].shape[1:], predictions[:, :4], shape, shapes[i][1])  # native-space pred

            # Evaluate
            if labels.shape[0]:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(images[i].shape[1:], tbox, shape, shapes[i][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predictions, labelsn, iouv)

            stats.append((correct, output[:, 4], output[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class

    # Print results
    print_format = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(print_format % ("all", seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(print_format % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="data/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, default="weights/weights/last.pt", help="model.pt path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")

    opt = parser.parse_args()
    LOGGER.info(colorstr("val: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
    return opt


def main(opt):
    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.")
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
