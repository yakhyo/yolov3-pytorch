# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Train a  model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov3.pt --img 640
"""
import argparse
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import val  # for end-of-epoch mAP
import yaml

from yolov3.models import YOLOv3SPP
from yolov3.models.common import ModelEMA
from tqdm import tqdm
from yolov3.utils import LOGGER
from yolov3.utils.datasets import create_dataloader
from yolov3.utils.general import check_img_size, colorstr, init_seeds, strip_optimizer
from yolov3.utils.loss import ComputeLoss
from yolov3.utils.metrics import fitness
from yolov3.utils.general import smart_optimizer
from yolov3.utils.misc import check_anchors, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def one_cycle(hyp, epochs):
    def fn(x):
        return ((1 - math.cos(x * math.pi / epochs)) / 2) * (hyp["lrf"] - 1) + 1  # linear

    return fn


def linear_lr(hyp, epochs):
    def fn(x):
        return (1 - x / (epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # cosine

    return fn


def train(hyp, opt, device):
    # path/to/hyp.yaml or hyp dictionary
    epochs, batch_size, data, resume, noval, nosave, workers = (
        opt.epochs,
        opt.batch_size,
        opt.data,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
    )

    # Directories
    w = "weights"  # weights dir
    os.makedirs(w, exist_ok=True)
    last, best = f"{w}/last.pt", f"{w}/best.pt"

    # Hyperparameters
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{param}={value}" for param, value in hyp.items()))

    # Config
    cuda = device.type != "cpu"
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        # Read yaml
        if isinstance(data, (str, Path)):
            with open(data, errors="ignore") as f:
                data = yaml.safe_load(f)  # dictionary

        # Parse yaml
        path = Path(data.get("path"))  # optional 'path' default to '.'
        for k in "train", "val", "test":
            if data.get(k):  # prepend path
                data[k] = str(path / data[k])

    data_dict = data
    train_path, val_path = data_dict["train"], data_dict["val"]

    nc = int(data_dict["nc"])  # number of classes
    names = data_dict["names"]  # class names

    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {data}"  # check

    # Model
    model = YOLOv3SPP(in_ch=3, num_classes=nc).to(device)
    pretrained = opt.weight.endswith(".pt")
    if pretrained:
        checkpoint = torch.load(opt.weight, map_location=device)
        state_dict = checkpoint["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
        LOGGER.info(f"Transferred {len(state_dict)}/{len(model.state_dict())} items from {opt.weight}")  # report

    # Image size
    grid_size = max(int(model.detect.stride.max()), 32)  # grid size (max stride)
    image_size = check_img_size(opt.image_size, grid_size, floor=grid_size * 2)  # verify image_size is gs-multiple

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    lf = linear_lr(hyp, epochs) if opt.linear_lr else one_cycle(hyp, epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_fitness = checkpoint["best_fitness"]

        # EMA
        if checkpoint.get("model"):
            ema.model.load_state_dict(checkpoint["model"].float().state_dict())
            ema.updates = checkpoint["updates"]

        # Epochs
        start_epoch = checkpoint["epoch"] + 1
        if resume:
            assert start_epoch > 0, f"{opt.weight} training to {epochs} epochs is finished, nothing to resume."
        if epochs < start_epoch:
            LOGGER.info(
                f"{opt.weight} has been trained for {checkpoint['epoch']} epochs. Fine-tuning for {epochs} more epochs."
            )
            epochs += checkpoint["epoch"]  # finetune additional epochs
        del checkpoint

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning("WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results")
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        image_size,
        batch_size // WORLD_SIZE,
        grid_size,
        hyp=hyp,
        augment=True,
        rank=LOCAL_RANK,
        workers=workers,
        prefix=colorstr("train: "),
        shuffle=True,
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in [-1, 0]:
        val_loader, _ = create_dataloader(
            val_path,
            image_size,
            batch_size // WORLD_SIZE * 2,
            grid_size,
            hyp=hyp,
            rank=-1,
            workers=workers,
            prefix=colorstr("val: "),
        )

        if not resume:
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=image_size)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and RANK != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    nl = model.module.detect.nl if hasattr(model, "module") else model.detect.nl  # number of detection layers
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / len(names) * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (image_size / 640) ** 2 * 3 / nl  # scale to image size and layers

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names  # attach class names to model

    # Start training
    num_warmup_iters = max(round(hyp["warmup_epochs"] * nb), 1000)  # number of warmup iters, max(3 epochs, 1k iters)
    last_opt_step = -1

    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class

    LOGGER.info(
        f"Image sizes {image_size} train, {image_size} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr(opt.save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()

        mean_loss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%10s" * 7) % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size"))
        if RANK in [-1, 0]:
            pbar = tqdm(
                pbar,
                total=nb,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )  # progress bar
        optimizer.zero_grad()
        # batch -------------------------------------------------------------
        for i, (imgs, targets, paths, _,) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= num_warmup_iters:
                xi = [0, num_warmup_iters]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Forward
            with torch.cuda.amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mean_loss = (mean_loss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%10s" * 2 + "%10.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mean_loss, targets.shape[0], imgs.shape[-1])
                )
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride"])
            final_epoch = (epoch + 1 == epochs)
            if not noval or final_epoch:  # Calculate mAP
                results, maps = val.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=image_size,
                    model=ema.model,
                    dataloader=val_loader,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            print("best_fitness: ", best_fitness)
            print("fi score: ", fi)
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if not nosave:  # if save
                checkpoint = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(ema.model).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                }

                # Save last, best and delete
                torch.save(checkpoint, last)
                if best_fitness == fi:
                    torch.save(checkpoint, best)

                del checkpoint

            # Stop Single-GPU
            if RANK == -1:
                break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        strip_optimizer(best)  # strip optimizers
        strip_optimizer(last)

    torch.cuda.empty_cache()
    return results


def attempt_load(weights, map_location=None):
    ckpt = torch.load(weights, map_location=map_location)  # load

    return ckpt["model"].float().eval()  # return model


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="./weights/last.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default="./data/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="./configs/hyp.scratch.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--image-size", "--img", "--img-size", type=int, default=640, help="train, val image size")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable autoanchor check")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=16, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    if RANK in [-1, 0]:
        LOGGER.info(colorstr("train: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))

    opt.weight = str(opt.weight)  # checks

    opt.save_dir = "weights"

    # DDP mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        assert opt.batch_size % WORLD_SIZE == 0, "--batch-size must be multiple of CUDA device count"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    train(opt.hyp, opt, device)
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info("Destroying process group... ")
        dist.destroy_process_group()


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', image_size=320, weight='yolov3.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
