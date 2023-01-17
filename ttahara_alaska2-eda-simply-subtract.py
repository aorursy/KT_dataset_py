import os

import sys

import gc



from pathlib import Path, PosixPath



import numpy as np

import pandas as pd



from PIL import Image, ImageChops



from matplotlib import pyplot as plt

import seaborn as sns



%matplotlib inline



ROOT = Path(".").resolve().parents[0]

INPUT_ROOT = ROOT / "input"

RAW_DATA = INPUT_ROOT / "alaska2-image-steganalysis"



TRAIN_COVER = RAW_DATA / "Cover"

TRAIN_JMiPOD = RAW_DATA / "JMiPOD"

TRAIN_JUNIWARD = RAW_DATA / "JUNIWARD"

TRAIN_UERD = RAW_DATA / "UERD"

TEST = RAW_DATA / "Test"
def read_image(image_id: str, image_dir: PosixPath):

    with open(image_dir / image_id, "rb") as fr:

        img = Image.open(fr)

        img.load()

    return img



def compare_image(image_id: str):

    cover = read_image(image_id, TRAIN_COVER)

    steganography = [

        ["JMiPOD", read_image(image_id, TRAIN_JMiPOD)],

        ["JUNIWARD", read_image(image_id, TRAIN_JUNIWARD)],

        ["UERD", read_image(image_id, TRAIN_UERD)],

    ]

    fig = plt.figure(figsize=(24, 27))

    for i, (name, ste_img) in enumerate(steganography):

        ax_cov = fig.add_subplot(3, 3, 3 * i + 1)

        ax_cov.set_title("Cover", fontsize=22)

        ax_ste = fig.add_subplot(3, 3, 3 * i + 2)

        ax_ste.set_title(name, fontsize=22)

        ax_sub = fig.add_subplot(3, 3, 3 * i + 3)

        ax_sub.set_title("SUB(Cover, {})".format(name), fontsize=22)

        

        ax_cov.imshow(cover)

        ax_ste.imshow(ste_img)

        sub_arr = np.asarray(cover) - np.asarray(ste_img)

        ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

        



def compare_crop_image(image_id: str, crop_area):

    cover = read_image(image_id, TRAIN_COVER)

    cover = cover.crop(crop_area)

    steganography = [

        ["JMiPOD", read_image(image_id, TRAIN_JMiPOD)],

        ["JUNIWARD", read_image(image_id, TRAIN_JUNIWARD)],

        ["UERD", read_image(image_id, TRAIN_UERD)],

    ]

    fig = plt.figure(figsize=(24, 27))

    for i, (name, ste_img) in enumerate(steganography):

        ste_img = ste_img.crop(crop_area)

        ax_cov = fig.add_subplot(3, 3, 3 * i + 1)

        ax_cov.set_title("Cover", fontsize=22)

        ax_ste = fig.add_subplot(3, 3, 3 * i + 2)

        ax_ste.set_title(name, fontsize=22)

        ax_sub = fig.add_subplot(3, 3, 3 * i + 3)

        ax_sub.set_title("SUB(Cover, {})".format(name), fontsize=22)

        

        ax_cov.imshow(cover)

        ax_ste.imshow(ste_img)

        sub_arr = np.asarray(cover) - np.asarray(ste_img)

        ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))
train_image_ids = sorted(os.listdir(TRAIN_COVER))

train_image_ids[:10]
compare_image(train_image_ids[0])
compare_image(train_image_ids[1])
compare_image(train_image_ids[2])
compare_image(train_image_ids[3])
compare_image(train_image_ids[4])
compare_image(train_image_ids[5])
compare_image(train_image_ids[6])
compare_crop_image(train_image_ids[0], (0, 0, 40, 40))
compare_crop_image(train_image_ids[1], (0, 0, 40, 40))
compare_crop_image(train_image_ids[2], (0, 0, 40, 40))
compare_crop_image(train_image_ids[3], (0, 0, 40, 40))
compare_crop_image(train_image_ids[4], (0, 0, 40, 40))
compare_crop_image(train_image_ids[5], (0, 0, 40, 40))
compare_crop_image(train_image_ids[6], (0, 0, 40, 40))