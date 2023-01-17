from fastai.vision import open_image
img_id = "0002cc93b"

img_path = f"../input/severstal-steel-defect-detection/train_images/{img_id}.jpg"
open_image(img_path)
# One channel from RGB but not necessary here (since all three channels are the same).

open_image(img_path, convert_mode = "L")
img = open_image(img_path)
# Displaying the first 5 * 5 patch

img.px[:, :5, :5]
# The image shape: (channels, width, height)

img.px.shape
# Notice that all channels contain the same information here.

print((img.px[0, : , :] == img.px[1, :, :]).all())

print((img.px[1, : , :] == img.px[2, :, :]).all())
import pandas as pd

train_df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")
train_df.head()
# Will extract img_id and class_id

train_df["img_id"] = train_df["ImageId_ClassId"].str.split(".").str[0]

train_df["class_id"] =  train_df["ImageId_ClassId"].str.split(".").str[1].str.split('_').str[1]
train_df.head()
train_df.loc[lambda df: df["img_id"] == img_id, ["EncodedPixels", "class_id"]]
# The mask is the first elemen

mask_rle = train_df.loc[lambda df: df["img_id"] == img_id, "EncodedPixels"].values[0]
from fastai.vision import open_mask_rle

mask_shape = (img.px.shape[1], img.px.shape[2])

mask = open_mask_rle(mask_rle, shape=mask_shape)

mask
from fastai.vision import ImageSegment

# Need to create a mask using the ImageSegment class

mask = ImageSegment(mask.data.transpose(2, 1))

mask
img.show(y=mask, figsize=(20, 10), title=f"{img_id} with mask, label 1")
!mkdir ../masks
import math

import torch

from fastai.vision import open_mask_rle, ImageSegment



def get_and_save_mask(img_id, df, shape=(1600, 256)):

    """ Extract the mask(s) for each image. The mask could be None."""

    # Shape: (width, height)

    # One mask (or none) per image.

    masks = []

    rle_df = df.loc[df["img_id"] == img_id, ['class_id', 'EncodedPixels']]

    # Not all images have masks

    for row in rle_df.itertuples():

        rle = row.EncodedPixels

        class_id = row.class_id

        if isinstance(rle, float) and math.isnan(rle):

            continue

        one_mask = open_mask_rle(rle, shape=shape)

        one_mask = int(class_id) * one_mask.data

        masks.append(one_mask)

    if len(masks) == 0:

        return

    stacked_mask = torch.stack(masks, dim=0).sum(dim=0)

    mask_img = ImageSegment(stacked_mask.reshape((1, shape[0], shape[1])).transpose(2, 1))

    mask_img.save(f"../masks/{img_id}.jpg")
# Run over all the train images

for img_id in train_df["img_id"].unique():

    get_and_save_mask(img_id, train_df)
from fastai.vision import open_mask

open_mask("../masks/0025bde0c.jpg")
# Filtering images with at least one mask

with_masks_df = (train_df.groupby('img_id')['EncodedPixels'].count() 

                      .reset_index()

                      .rename(columns={"EncodedPixels": "n_masks"}))

with_masks_df = with_masks_df.loc[lambda df: df["n_masks"] > 0, :]
# data pipeline

from fastai.vision import SegmentationItemList, get_transforms



train_folder = "../input/severstal-steel-defect-detection/train_images/"

sl = SegmentationItemList.from_df(with_masks_df, train_folder, suffix=".jpg")

size = 256

batch_size = 16

data = (sl.split_none()

          .label_from_func(lambda x : str(x).replace(train_folder, '../masks/'),

                           classes=[0, 1, 2, 3, 4])

          .transform(get_transforms(), size=size, tfm_y=True)

          .databunch(bs=batch_size))
data.show_batch()