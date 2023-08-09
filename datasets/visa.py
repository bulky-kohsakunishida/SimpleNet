import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

from pathlib import Path

import csv
import logging
import shutil

import cv2

logger = logging.getLogger("simplenet")

_CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.apply_cls1_split()

        self.new_source = os.path.join(source, "visa_pytorch")

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.new_source, classname, self.split.value)
            maskpath = os.path.join(self.new_source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.split_root / self.category).is_dir():
            # dataset is available, and split has been applied
            logger.info("Found the dataset and train/test split.")
        elif (self.root / self.category).is_dir():
            # dataset is available, but split has not yet been applied
            logger.info("Found the dataset. Applying train/test split.")
            self.apply_cls1_split()
        else:
            # dataset is not available
            download_and_extract(self.root, DOWNLOAD_INFO)
            logger.info("Downloaded the dataset. Applying train/test split.")
            self.apply_cls1_split()

    def apply_cls1_split(self) -> None:
        """Apply the 1-class subset splitting using the fixed split in the csv file.

        adapted from https://github.com/amazon-science/spot-diff
        """
        logger.info("preparing data")
        categories = [
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ]

        print("apply_class")

        split_file = Path(self.source) / "split_csv" / "1cls.csv"

        self.split_root = Path(self.source) / "visa_pytorch"

        for category in categories:
            train_folder = self.split_root / category / "train"
            test_folder = self.split_root / category / "test"
            mask_folder = self.split_root / category / "ground_truth"

            train_img_good_folder = train_folder / "good"
            test_img_good_folder = test_folder / "good"
            test_img_bad_folder = test_folder / "bad"
            test_mask_bad_folder = mask_folder / "bad"

            train_img_good_folder.mkdir(parents=True, exist_ok=True)
            test_img_good_folder.mkdir(parents=True, exist_ok=True)
            test_img_bad_folder.mkdir(parents=True, exist_ok=True)
            test_mask_bad_folder.mkdir(parents=True, exist_ok=True)

        with split_file.open(encoding="utf-8") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                category, split, label, image_path, mask_path = row
                if label == "normal":
                    label = "good"
                else:
                    label = "bad"
                image_name = image_path.split("/")[-1]
                mask_name = mask_path.split("/")[-1]

                img_src_path = Path(self.source)/ image_path
                msk_src_path = Path(self.source) / mask_path
                img_dst_path = self.split_root / category / split / label / image_name
                msk_dst_path = self.split_root / category / "ground_truth" / label / mask_name

                shutil.copyfile(img_src_path, img_dst_path)
                if split == "test" and label == "bad":
                    mask = cv2.imread(str(msk_src_path))

                    # binarize mask
                    mask[mask != 0] = 255

                    cv2.imwrite(str(msk_dst_path), mask)
