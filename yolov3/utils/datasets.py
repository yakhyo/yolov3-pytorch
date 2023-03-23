import os
import random
from pathlib import Path

import cv2
import numpy as np

import torch
from PIL import Image
from torch.utils import data

from tqdm import tqdm

from yolov3 import LOGGER
from yolov3.utils.augmentations import augment_hsv, letterbox, mixup, random_perspective, xywhn2xyxy, xyxy2xywhn
from yolov3.utils.misc import torch_distributed_zero_first

# Parameters
IMG_FORMATS = ["jpg", "jpeg", "png"]
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # DPP


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
    num_workers = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return (
        torch.utils.data.DataLoader(
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


class LoadImagesAndLabels(torch.utils.data.Dataset):
    #  train_loader/val_loader, loads images and labels for training and validation

    def __init__(self, path, image_size=640, augment=False, hyp=None, stride=32, prefix=""):
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
            p = Path(path)  # os-agnostic
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
            self.image_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.image_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}")
        # Check cache
        self.label_files = image2label(self.image_files)  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (
                np.load(str(cache_path), allow_pickle=True).item(),
                True,
            )  # load dict
        except (FileNotFoundError, AssertionError):
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            LOGGER.info(f"{prefix}Scanning '{cache_path}' {nf} found, {nm} missing, {ne} empty, {nc} corrupted")
        assert nf > 0 or not augment, f"{prefix}No labels in {cache_path}. Can not train without labels."

        # Read cache
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
        index = self.indices[idx]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            shapes = None
            image, labels = self.load_mosaic(index, hyp)
            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                random_idx = random.randint(0, self.num - 1)
                image, labels = mixup(image, labels, *self.load_mosaic(random_idx))

        else:
            # Load image
            image, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            image, ratio, pad = letterbox(image, self.input_size, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                image, labels = random_perspective(image, labels, hyp)

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

        targets = torch.zeros((nl, 6))
        if nl:
            targets[:, 1:] = torch.from_numpy(labels)

        # Convert HWC to CHW, BGR to RGB
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image), targets, self.image_files[idx], shapes

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
                assert image.format.lower() in IMG_FORMATS, f"invalid image format {image.format}"
                # verify labels
                if os.path.isfile(label_file):
                    nf += 1  # label found
                    with open(label_file) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, f"labels require 5 columns, {label.shape[1]} columns detected"
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

        x["results"] = nf, nm, ne, nc, len(self.image_files)

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
        return torch.stack(img), torch.cat(label), path, shapes

    def load_image(self, idx):
        image = cv2.imread(self.image_files[idx])  # BGR
        assert image is not None, f"Image Not Found {self.image_files[idx]}"
        h, w = image.shape[:2]  # orig hw
        r = self.input_size / max(h, w)  # ratio
        if r != 1:
            interpolation_mode = cv2.INTER_AREA if (r < 1 and not self.augment) else cv2.INTER_LINEAR
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=interpolation_mode)
        return image, (h, w), image.shape[:2]  # image, hw_original, hw_resized

    def load_mosaic(self, idx, hyp):
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
            hyp,
            border=self.mosaic_border,
        )  # border to remove

        return image4, labels4
