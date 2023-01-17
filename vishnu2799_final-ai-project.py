import numpy as np # basic linear algebra and array operations

import pandas as pd # used for data processing,eg for I/O of a CSV file. 

import os  # for the path operations

print(os.listdir("../input"))

import cv2   # cv2 is used for image processing operations

import matplotlib.pyplot as plt  # matplotlib is used for data visualization

%matplotlib inline

from tqdm import tqdm_notebook as tqdm  # tqdm is used for visualizing the progress of running



import torch # Contains data structures for multi-dimensional tensors and mathematical operations.

import torch.nn as nn # A kind of Tensor, to be considered a module parameter in torch

from torch import optim # A package implementing various optimization algorithms

import torchvision.transforms as transforms  # Used for common image transformations.

import torch.nn.functional as F  # Used for Convolution Functions

from torch.autograd import Function, Variable  # Automatic differentiation of arbitrary scalar valued functions.

from pathlib import Path # For getting the path 

from itertools import groupby # To group the elements
input_dir = "../input/imaterialist-fashion-2019-FGVC6/"

train_img_dir = "../input/imaterialist-fashion-2019-FGVC6/train/"

test_img_dir = "../input/imaterialist-fashion-2019-FGVC6/test/"



WIDTH = 512

HEIGHT = 512

category_num = 47



ratio = 8



epoch_num = 2

batch_size = 4



device = "cuda:0"
len(os.listdir("../input/imaterialist-fashion-2019-FGVC6/train/"))
len(os.listdir(test_img_dir))
train_df = pd.read_csv(input_dir + "train.csv")

train_df.head()
train_df.shape
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')



!rm -rf .git # To prevent an error when the kernel is committed

!rm -rf images assets # To prevent displaying images at the bottom of a kernel
import sys

sys.path.append(str('/kaggle/working/Mask_RCNN'))

from mrcnn.config import Config

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

!ls -lh mask_rcnn_coco.h5



COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
# For demonstration purpose, the classification ignores attributes (only categories),

# and the image size is set to 512, which is the same as the size of submission masks



NUM_CATS = 46

IMAGE_SIZE = 512
class FashionConfig(Config):

    NAME = "fashion"

    NUM_CLASSES = NUM_CATS + 1 

    

    GPU_COUNT = 1

    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high

    

    BACKBONE = 'resnet50'

    

    IMAGE_MIN_DIM = IMAGE_SIZE

    IMAGE_MAX_DIM = IMAGE_SIZE    

    IMAGE_RESIZE_MODE = 'none'

    

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 200

    

config = FashionConfig()

config.display()
import json

with open("/kaggle/input/imaterialist-fashion-2019-FGVC6/label_descriptions.json") as f:

    label_descriptions = json.load(f)



label_names = [x['name'] for x in label_descriptions['categories']]
print(label_names)
print(len(label_names))
attribute_names = [x['name'] for x in label_descriptions['attributes']]
print(attribute_names)
print(len(attribute_names))
print(label_descriptions)
segment_df = pd.read_csv("/kaggle/input/imaterialist-fashion-2019-FGVC6/train.csv")



multilabel_percent = len(segment_df[segment_df['ClassId'].str.contains('_')])/len(segment_df)*100

print(f"Segments that have attributes: {multilabel_percent:.2f}%")
segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]



print("Total segments: ", len(segment_df))

segment_df.head()
image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))

size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()

image_df = image_df.join(size_df, on='ImageId')



print("Total images: ", len(image_df))

image_df.head()
def resize_image(image_path):

    try:

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    except Exception as e:

        pass

    return img
from pathlib import Path

DATA_DIR = Path('/kaggle/input/imaterialist-fashion-2019-FGVC6')

ROOT_DIR = Path('/kaggle/working/imaterialist-fashion-2019-FGVC6')
class FashionDataset(utils.Dataset):

    def __init__(self, df):

        super().__init__(self)

        

        # Adding the  classes

        for i, name in enumerate(label_names):

            self.add_class("fashion", i+1, name)

        

        # Add the images 

        for i, row in df.iterrows():

            self.add_image("fashion", image_id=row.name, path=str(DATA_DIR/'train'/row.name), labels=row['CategoryId'],

                           annotations=row['EncodedPixels'], height=row['Height'], width=row['Width'])

    

    # This function returns the path and the label_names of the image

    def image_reference(self, image_id):

        info = self.image_info[image_id]

        return info['path'], [label_names[int(x)] for x in info['labels']]

    

    # This function is used to resize the image

    def load_image(self, image_id):

        return resize_image(self.image_info[image_id]['path'])



    # This function is used to generate a mask for the given image

    def load_mask(self, image_id):

        info = self.image_info[image_id]

                

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)

        labels = []

        

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):

            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)

            annotation = [int(x) for x in annotation.split(' ')]

            

            for i, start_pixel in enumerate(annotation[::2]):

                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1



            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')

            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)            

            mask[:, :, m] = sub_mask

            labels.append(int(label)+1)

            

        return mask, np.array(labels)
import random

dataset = FashionDataset(image_df)

dataset.prepare()



for i in range(6):

    image_id = random.choice(dataset.image_ids)

    print(dataset.image_reference(image_id))

    

    image = dataset.load_image(image_id)

    mask, class_ids = dataset.load_mask(image_id)

    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
# Used to perform one hot encoding of the categorical variables.



def make_onehot_vec(x):

    vec = np.zeros(category_num)

    vec[x] = 1

    return vec
# This function is used to create a mask of the costumes which are in the image dataset

def make_mask_img(segment_df):

    seg_width = segment_df.at[0, "Width"]

    seg_height = segment_df.at[0, "Height"]

    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)

    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

        pixel_list = list(map(int, encoded_pixels.split(" ")))

        for i in range(0, len(pixel_list), 2):

            start_index = pixel_list[i] - 1

            index_len = pixel_list[i+1] - 1

            seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])

    seg_img = seg_img.reshape((seg_height, seg_width), order='F')

    seg_img = cv2.resize(seg_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    return seg_img
# This Utility function is used to generate the training dataset in a format which can be used while giving the 

# dataset to the traning model

def train_generator(df, batch_size):

    img_ind_num = df.groupby("ImageId")["ClassId"].count() 

    index = df.index.values[0]

    trn_images = []

    seg_images = []

    for i, (img_name, ind_num) in enumerate(img_ind_num.items()):

        try:

            img = cv2.imread(train_img_dir + img_name)

            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            segment_df = (df.loc[index:index+ind_num-1, :]).reset_index(drop=True)

            index += ind_num

            if segment_df["ImageId"].nunique() != 1:

                raise Exception("Index Range Error")

            seg_img = make_mask_img(segment_df)

        

            img = img.transpose((2, 0, 1))    

            trn_images.append(img)

            seg_images.append(seg_img)

            if((i+1) % batch_size == 0):

                yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)

                trn_images = []

                seg_images = []

        except Exception as e:

            pass

        if(len(trn_images) != 0):

            yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)

        
# This Utility function is used to generate the test dataset in a format which is same as the train dataset

def test_generator(df):

    img_names = df["ImageId"].values

    for img_name in img_names:

        try:

            img = cv2.imread(test_img_dir + img_name)

            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            img = img.transpose((2, 0, 1))

        except Exception as e:

            pass

        yield img_name, np.asarray([img], dtype=np.float32) / 255

# This Utility function is used to encode the string

def encode(input_string):

    return [(len(list(g)), k) for k,g in groupby(input_string)]



# This Utility function is used to perform run length encoding

def run_length(label_vec):

    encode_list = encode(label_vec)

    index = 1

    class_dict = {}

    for i in encode_list:

        if i[1] != category_num-1:

            if i[1] not in class_dict.keys():

                class_dict[i[1]] = []

            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]

        index += i[0]

    return class_dict
class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(double_conv, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))



    def forward(self, x):

        x = self.conv(x)

        return x





class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x):

        x = self.conv(x)

        return x





class down(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(down, self).__init__()

        self.mpconv = nn.Sequential(nn.MaxPool2d(2),double_conv(in_ch, out_ch))



    def forward(self, x):

        x = self.mpconv(x)

        return x





class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        super(up, self).__init__()



        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:

            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)



        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffX = x1.size()[2] - x2.size()[2]

        diffY = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x





class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(outconv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1)



    def forward(self, x):

        x = self.conv(x)

        return x



    

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):

        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)

        self.down1 = down(64, 128)

        self.down2 = down(128, 256)

        self.down3 = down(256, 512)

        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)

        self.up2 = up(512, 128)

        self.up3 = up(256, 64)

        self.up4 = up(128, 64)

        self.outc = outconv(64, n_classes)



    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.outc(x)

        return x
train_df.shape
333415 // 4  # Since it is 4 batches
train_df.iloc[83348:83354, :]  # We will have a look at a small part of the dataset
train_df.iloc[73350:73354, :]
net = UNet(n_channels=3, n_classes=category_num).to(device)  #Trains a unet instance

optimizer = optim.SGD(net.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005) # Optimizing the algorithm

criterion = nn.CrossEntropyLoss()  # It is useful when training a classification problem with a particular number of classes.
val_sta = 73352

val_end = 83351

train_loss = []

valid_loss = []

for epoch in range(epoch_num):

    epoch_trn_loss = 0

    train_len = 0

    net.train()

    

    # This is for training dataset

    for iteration, (X_trn, Y_trn) in enumerate(tqdm(train_generator(train_df.iloc[:val_sta, :], batch_size))):

        X = torch.tensor(X_trn, dtype=torch.float32).to(device)  #torch.Tensor is a multi-dimensional matrix containing elements of a single data type.

        Y = torch.tensor(Y_trn, dtype=torch.long).to(device)

        train_len += len(X)

        

        mask_pred = net(X)

        loss = criterion(mask_pred, Y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_trn_loss += loss.item()

        

        if iteration % 100 == 0:

            print("train loss in {:0>2}epoch  /{:>5}iter:{:<10.8}".format(epoch+1, iteration, epoch_trn_loss/(iteration+1)))

        

    train_loss.append(epoch_trn_loss/(iteration+1))

    print("train {}epoch loss({}iteration):{:10.8}".format(epoch+1, iteration, train_loss[-1]))

    

    

    # This is for validation dataset

    epoch_val_loss = 0

    val_len = 0

    net.eval()

    for iteration, (X_val, Y_val) in enumerate(tqdm(train_generator(train_df.iloc[val_sta:val_end, :], batch_size))):

        X = torch.tensor(X_val, dtype=torch.float32).to(device)

        Y = torch.tensor(Y_val, dtype=torch.long).to(device)

        val_len += len(X)

            

        mask_pred = net(X)

        loss = criterion(mask_pred, Y)

        epoch_val_loss += loss.item()

        

        if iteration % 100 == 0:

            print("valid loss in {:0>2}epoch/{:>5}iter: {:<10.8}".format(epoch+1, iteration, epoch_val_loss/(iteration+1)))

        

    valid_loss.append(epoch_val_loss/(iteration+1))

    print("valid {}epoch loss({}iteration): {:10.8}".format(epoch+1, iteration, valid_loss[-1]))
sample_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2019-FGVC6/sample_submission.csv')

sample_df.head()
# Generating the predictions in the sub_list list

sub_list = []

net.eval()

for img_name, img in test_generator(sample_df):

    X = torch.tensor(img, dtype=torch.float32).to(device)

    mask_pred = net(X)

    mask_pred = mask_pred.cpu().detach().numpy()

    mask_prob = np.argmax(mask_pred, axis=1)

    mask_prob = mask_prob.ravel(order='F')

    class_dict = run_length(mask_prob)

    if len(class_dict) == 0:

        sub_list.append([img_name, "1 1", 1])

    else:

        for key, val in class_dict.items():

            sub_list.append([img_name, " ".join(map(str, val)), key])
submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
submission_df.head()