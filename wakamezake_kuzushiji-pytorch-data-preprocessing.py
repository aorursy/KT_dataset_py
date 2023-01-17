%ls ../input
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path



import torch

import torch.nn.functional as F

from torch import nn

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import torchvision

import torchvision.transforms as transforms

from torch.autograd import Variable

input_path = Path("../input")

%matplotlib inline
# reference

# https://www.kaggle.com/anokas/kuzushiji-visualisation

df_train = pd.read_csv( input_path / 'train.csv')

unicode_trans = pd.read_csv( input_path / 'unicode_translation.csv')

train_image_path = input_path / "train_images"

test_image_path = input_path / "test_images"

unicode_map = {codepoint: char for codepoint, char in unicode_trans.values}
df_train.head()
length = 5

split_labels = df_train["labels"][0].split()

for idx in range(len(split_labels) // length):

    start_idx = idx * length

    print(split_labels[start_idx : start_idx + length])

    if idx == 4:

        break
class KuzushijiDataset(Dataset):

    def __init__(self, img_path, mode="train"):

        self.img_path = img_path

        self.img_paths = list(self.img_path.glob("*jpg"))

        self.unicode_trans = pd.read_csv( input_path / 'unicode_translation.csv')

        self.unicode_map = {codepoint: char for codepoint, char in unicode_trans.values}

        self.unicode2labels = dict(zip(self.unicode_map.keys(),

                                      range(len(self.unicode_map.keys()))))

        self.label_length = 5

        self.transform = transforms.ToTensor()

        if mode == "train":

            self.mode = "train"

            self.mask = pd.read_csv( input_path / 'train.csv')

        else:

            self.mode = "test"

    

    def get_label_and_mask(self, image_id):

#         assert type(image_id) == str

        split_labels = self.mask[self.mask["image_id"] == image_id]["labels"].str.split(" ").values[0]

        ll = len(split_labels) // length

        masks = np.zeros((ll, 4))

        labels = np.zeros((ll))

        for idx in range(ll):

            start_idx = idx * self.label_length

            labels[idx] = self.unicode2labels[split_labels[start_idx]]

            masks[idx] = split_labels[start_idx+1:start_idx+self.label_length]

        return labels, masks

    

    def __getitem__(self, index):

        """ Get a sample from the dataset

        """

        img_path = self.img_paths[index]

        image = Image.open(img_path)

        if self.mode == "train":

            labels, masks = self.get_label_and_mask(img_path.stem)

#             data = {"image": np.array(image), "mask": masks}

#         else:

#             data = {"image": np.array(image)}

#         transformed = self.transform(**data)

#         image = transformed['image'] / 255

#         image = np.transpose(image, (2, 0, 1))

        if self.mode == 'train':

            return self.transform(image), labels, masks

        else:

            return image



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return len(self.img_paths)
k_train = KuzushijiDataset(img_path=train_image_path)
# reference

# https://www.kaggle.com/anokas/kuzushiji-visualisation

# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated

def visualize_training_data(image_fn, masks):

    imsource = Image.fromarray(np.uint8(image_fn * 255)).convert('RGBA')

    bbox_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character



    for idx in range(masks.shape[1]):

        x, y, w, h = masks[0][idx]

        x, y, w, h = int(x), int(y), int(w), int(h)



        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x, y, x+w, y+h),

                            fill=(255, 255, 255, 0),

                            outline=(255, 0, 0, 255))

    imsource = Image.alpha_composite(imsource, bbox_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
batch_size = 1



# Use the torch dataloader to iterate through the dataset

loader = DataLoader(k_train, batch_size=batch_size, shuffle=False, num_workers=0)



# functions to show an image

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))



# get some images

dataiter = iter(loader)

images, label, masks  = dataiter.next()

print("image.shape: {}".format(images.shape))

print("mask.shape: {}".format(masks.shape))



# show images

plt.figure(figsize=(12,12))

np_img = images.numpy()[0].transpose((1, 2, 0))

new_img = visualize_training_data(np_img, masks)

plt.imshow(new_img, interpolation='lanczos')
images, label, masks  = dataiter.next()

print("image.shape: {}".format(images.shape))

print("mask.shape: {}".format(masks.shape))



# show images

plt.figure(figsize=(12,12))

np_img = images.numpy()[0].transpose((1, 2, 0))

new_img = visualize_training_data(np_img, masks)

plt.imshow(new_img, interpolation='lanczos')
images, label, masks  = dataiter.next()

print("image.shape: {}".format(images.shape))

print("mask.shape: {}".format(masks.shape))



# show images

plt.figure(figsize=(12,12))

np_img = images.numpy()[0].transpose((1, 2, 0))

new_img = visualize_training_data(np_img, masks)

plt.imshow(new_img, interpolation='lanczos')