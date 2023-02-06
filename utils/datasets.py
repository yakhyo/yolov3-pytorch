# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
import glob
import hashlib
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, distributed
from tqdm import tqdm

from utils import LOGGER
from utils.general import clip_coords
from utils.torch_utils import torch_distributed_zero_first

# Parameters
IMG_FORMATS = ["jpg", "jpeg", "png"]
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # DPP
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def create_dataloader(
        path,
        image_size,
        batch_size,
        stride,
        hyp=None,
        augment=False,
        rank=-1,
        workers=16,
        prefix="",
        shuffle=False,
):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            image_size=image_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            stride=int(stride),
            prefix=prefix,
        )

    batch_size = min(batch_size, len(dataset))
    num_workers = min(
        [os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn,
        ),
        dataset,
    )


def image2label(image_paths):
    # Define label paths as a function of image paths
    sa = os.sep + "images" + os.sep  # /labels/
    sb = os.sep + "labels" + os.sep  # /images/
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in image_paths]


class LoadImagesAndLabels(Dataset):
    #  train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(
            self, path, image_size=640, augment=False, hyp=None, stride=32, prefix=""
    ):
        self.input_size = image_size
        self.hyp = hyp
        self.mosaic = self.augment = augment
        self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.stride = stride

        # try:
        #     path = Path(path)
        #     assert path.is_dir(), f"{prefix}{path} does not exist"
        #     self.image_files = [os.path.join(path, x) for x in os.listdir(path) if x.split(".")[-1] in IMG_FORMATS]
        #     assert self.image_files, f"{prefix}No images found"
        # except Exception as e:
        #     raise Exception(f"{prefix}Error loading data from {path}: {e}")
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x
                            for x in t
                        ]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f"{prefix}{p} does not exist")
            self.image_files = sorted(
                x.replace("/", os.sep)
                for x in f
                if x.split(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.image_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}")
        # Check cache
        self.label_files = image2label(self.image_files)  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (np.load(str(cache_path), allow_pickle=True).item(), True,)  # load dict
            assert cache["version"] == self.cache_version  # same version
            assert cache["hash"] == get_hash(self.label_files + self.image_files)  # same hash
        except (FileNotFoundError, AssertionError):
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            LOGGER.info(f"{prefix}Scanning '{cache_path}' {nf} found, {nm} missing, {ne} empty, {nc} corrupted")
        assert (nf > 0 or not augment), f"{prefix}No labels in {cache_path}. Can not train without labels."

        # Read cache
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels, shapes = zip(*cache.values())

        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)

        self.num = len(labels)
        self.indices = range(self.num)

        self.image_files = list(cache.keys())  # update
        self.label_files = image2label(cache.keys())  # update

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        idx = self.indices[idx]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            image, labels = self.load_mosaic(idx)
            shapes = None
            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                image, labels = mixup(image, labels, *self.load_mosaic(random.randint(0, self.num - 1)))

        else:
            # Load image
            image, (h0, w0), (h, w) = self.load_image(idx)

            # Letterbox
            shape = self.input_size  # final letterboxed shape
            image, ratio, pad = letterbox(image, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            labels = self.labels[idx].copy()

            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                image, labels = random_perspective(
                    image,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=image.shape[1], h=image.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # HSV color-space
            augment_hsv(image, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                image = np.flipud(image)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                image = np.fliplr(image)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image), labels_out, self.image_files[idx], shapes

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = (0, 0, 0, 0, [])  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        pbar = tqdm(zip(self.image_files, self.label_files), total=len(self.image_files), desc=desc)
        for image_file, label_file in pbar:
            try:
                # verify images
                image = Image.open(image_file)
                image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                assert (image.format.lower() in IMG_FORMATS), f"invalid image format {image.format}"
                # verify labels
                if os.path.isfile(label_file):
                    nf += 1  # label found
                    with open(label_file) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert (label.shape[1] == 5), f"labels require 5 columns, {label.shape[1]} columns detected"
                        assert (label >= 0).all(), f"negative label values {label[label < 0]}"
                        assert (label[:, 1:] <= 1).all(), f"non-normalized {label[:, 1:][label[:, 1:] > 1]}"
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                            msgs.append(f"{prefix}WARNING: {image_file}: {nl - len(i)} duplicate labels removed")
                    else:
                        ne += 1  # label empty
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    label = np.zeros((0, 5), dtype=np.float32)

                if image_file:
                    x[image_file] = [label, shape]
            except Exception as e:
                nc += 1
                msgs.append(f"{prefix}WARNING: {image_file}: ignoring corrupt image/label: {e}")
        pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING: No labels found in {path}.")

        x["hash"] = get_hash(self.label_files + self.image_files)
        x["results"] = nf, nm, ne, nc, len(self.image_files)
        x["version"] = self.cache_version  # cache version

        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def load_image(self, idx):
        image = cv2.imread(self.image_files[idx])  # BGR
        assert image is not None, f"Image Not Found {self.image_files[idx]}"
        h, w = image.shape[:2]  # orig hw
        r = self.input_size / max(h, w)  # ratio
        if r != 1:
            resample = (cv2.INTER_AREA if (r < 1 and not self.augment) else cv2.INTER_LINEAR)
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=resample)
        return image, (h, w), image.shape[:2]  # image, hw_original, hw_resized

    def load_mosaic(self, idx):
        #  4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        image4 = np.full((self.input_size * 2, self.input_size * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = None, None, None, None, None, None, None, None

        yc = int(random.uniform(self.input_size // 2, 2 * self.input_size - self.input_size // 2))
        xc = int(random.uniform(self.input_size // 2, 2 * self.input_size - self.input_size // 2))

        indices = [idx] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            # Load image
            image, _, (h, w) = self.load_image(idx)
            # place image in image4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # x_min, y_min, x_max, y_max (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # x_min, y_min, x_max, y_max (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    xc,
                    max(yc - h, 0),
                    min(xc + w, self.input_size * 2),
                    yc,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    yc,
                    xc,
                    min(self.input_size * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    xc,
                    yc,
                    min(xc + w, self.input_size * 2),
                    min(self.input_size * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # image4[y_min:y_max, x_min:x_max]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            labels = self.labels[idx].copy()
            if len(labels):
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, pad_w, pad_h)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * self.input_size, out=x)  # clip when using random_perspective()

        # Augment
        image4, labels4 = random_perspective(
            image4,
            labels4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return image4, labels4


def random_perspective(
        image,
        targets=(),
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=10,
        perspective=0.0,
        border=(0, 0),
):
    # targets = [cls, xyxy]
    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return image, targets


def box_candidates(
        box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16
):  # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def mixup(images1, labels1, images2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    images1 = (images1 * r + images2 * (1 - r)).astype(np.uint8)
    labels1 = np.concatenate((labels1, labels2), 0)
    return images1, labels1


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y
