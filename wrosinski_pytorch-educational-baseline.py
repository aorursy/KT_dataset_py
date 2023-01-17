import gc
import glob
import os
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import models

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 10)
# Basic 2D Convolution with 3x3 kernel
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


# conv3x3 with ReLU activation afterwards
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


# Decoder block containing 2D transposed Convolution upsampling the features
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# UNet architecture based on VGG11
class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        """
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # VGG11 encoder
        self.encoder = models.vgg11().features

        # Encoder part
        self.relu = self.encoder[1]

        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        # Decoder part
        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        # Output layer
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # Deconvolutions with copies of VGG11 layers of corresponding size
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        
        # Sigmoid over final convolution map is needed for Binary Crossentropy loss
        output = F.sigmoid(self.final(dec1))
        
        return output


def get_model(params):
    model = UNet11(**params)
    model.train()  # set model for training
    return model.to(device)  # put model on selected device, CPU ('cpu') or GPU ('cuda')
class TGSSaltDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 file_list,
                 is_test=False,
                 divide=False,
                 image_size=(128, 128)):

        self.root_path = root_path
        self.file_list = file_list
        self.is_test = is_test

        self.divide = divide
        self.image_size = image_size

        self.orig_image_size = (101, 101)
        self.padding_pixels = None
        
        """
        root_path: folder specifying files location
        file_list: list of images IDs
        is_test: whether train or test data is used (contains masks or not)
        
        divide: whether to divide by 255
        image_size: output image size, should be divisible by 32
        
        orig_image_size: original images size
        padding_pixels: placeholder for list of padding dimensions
        """

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        # Get image path
        image_folder = os.path.join(self.root_path, 'images')
        image_path = os.path.join(image_folder, file_id + '.png')
    
        # Get mask path
        mask_folder = os.path.join(self.root_path, 'masks')
        mask_path = os.path.join(mask_folder, file_id + '.png')

        # Load image
        image = self.__load_image(image_path)
        if not self.is_test:
            # Load mask for training or evaluation
            mask = self.__load_image(mask_path, mask=True)
            if self.divide:
                image = image / 255.
                mask = mask / 255.
            # Transform into torch float Tensors of shape (CxHxW).
            image = torch.from_numpy(
                image).float().permute([2, 0, 1])
            mask = torch.from_numpy(
                np.expand_dims(mask, axis=-1)).float().permute([2, 0, 1])
            return image, mask

        if self.is_test:
            if self.divide:
                image = image / 255.
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            return (image,)

    def set_padding(self):

        """
        Compute padding borders for images based on original and specified image size.
        """
        
        pad_floor = np.floor(
            (np.asarray(self.image_size) - np.asarray(self.orig_image_size)) / 2)
        pad_ceil = np.ceil((np.asarray(self.image_size) -
                            np.asarray(self.orig_image_size)) / 2)

        self.padding_pixels = np.asarray(
            (pad_floor[0], pad_ceil[0], pad_floor[1], pad_ceil[1])).astype(np.int32)

        return

    def __pad_image(self, img):
        
        """
        Pad images according to border set in set_padding.
        Original image is centered.
        """

        y_min_pad, y_max_pad, x_min_pad, x_max_pad = self.padding_pixels[
            0], self.padding_pixels[1], self.padding_pixels[2], self.padding_pixels[3]

        img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad,
                                 x_min_pad, x_max_pad,
                                 cv2.BORDER_REPLICATE)

        assert img.shape[:2] == self.image_size, '\
        Image after padding must have the same shape as input image.'

        return img

    def __load_image(self, path, mask=False):
        
        """
        Helper function for loading image.
        If mask is loaded, it is loaded in grayscale (, 0) parameter.
        """

        if mask:
            img = cv2.imread(str(path), 0)
        else:
            img = cv2.imread(str(path))

        height, width = img.shape[0], img.shape[1]

        img = self.__pad_image(img)

        return img

    def return_padding_borders(self):
        """
        Return padding borders to easily crop the images.
        """
        return self.padding_pixels
device = 'cuda:0'
data_src = '../input/'

quick_try = False
grayscale = False

orig_image_size = (101, 101)
image_size = (128, 128)
print('Initialize.')

train_df = pd.read_csv('{}train.csv'.format(data_src),
                       usecols=[0], index_col='id')
depths_df = pd.read_csv('{}depths.csv'.format(data_src),
                        index_col='id')

train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
X_train = []
y_train = []

print('Loading training set.')
for i in tqdm.tqdm(train_df.index):
    img_src = '{}train/images/{}.png'.format(data_src, i)
    mask_src = '{}train/masks/{}.png'.format(data_src, i)
    if grayscale:
        img_temp = cv2.imread(img_src, 0)
    else:
        img_temp = cv2.imread(img_src)
    mask_temp = cv2.imread(mask_src, 0)
    if orig_image_size != image_size:
        img_temp = cv2.resize(img_temp, image_size)
        mask_temp = cv2.resize(mask_temp, image_size)
    X_train.append(img_temp)
    y_train.append(mask_temp)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
if grayscale:
    X_train = np.expand_dims(X_train, -1)
y_train = np.expand_dims(y_train, -1)
print('Compute mask coverage for each observation.')

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

# Percent of area covered by mask.
train_df['coverage'] = np.mean(y_train / 255., axis=(1, 2))
train_df['coverage_class'] = train_df.coverage.map(
    cov_to_class)


# del X_train, y_train
# gc.collect()
plt.imshow(y_train[-2, :, :, 0])
train_df.coverage_class
train_path = data_src + 'train'
test_path = data_src

train_ids = train_df.index.values
test_ids = test_df.index.values
from sklearn.model_selection import train_test_split

tr_ids, valid_ids, tr_coverage, valid_coverage = train_test_split(
    train_ids,
    train_df.coverage.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)
# Training dataset:
dataset_train = TGSSaltDataset(train_path, tr_ids, divide=True)
dataset_train.set_padding()
y_min_pad, y_max_pad, x_min_pad, x_max_pad = dataset_train.return_padding_borders()
        
# Validation dataset:
dataset_val = TGSSaltDataset(train_path, valid_ids, divide=True)
dataset_val.set_padding()

# Test dataset:
dataset_test = TGSSaltDataset(test_path, test_ids, is_test=True, divide=True)
dataset_test.set_padding()


# Data loaders:
# Use multiple workers to optimize data loading speed.
# Pin memory for quicker GPU processing.
train_loader = data.DataLoader(
    dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

# Do not shuffle for validation and test.
valid_loader = data.DataLoader(
    dataset_val,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

test_loader = data.DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
# Get defined UNet model.
model = get_model({'num_filters': 32})
# Set Binary Crossentropy as loss function.
loss_fn = torch.nn.BCELoss()

# Set optimizer.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train for n epochs
n = 2
for e in range(n):

    # Training:
    train_loss = []
    for image, mask in tqdm.tqdm(train_loader):

        # Put image on chosen device
        image = image.type(torch.float).to(device)
        # Predict with model:
        y_pred = model(image)
        # Compute loss between true and predicted values
        loss = loss_fn(y_pred, mask.to(device))

        # Set model gradients to zero.
        optimizer.zero_grad()
        # Backpropagate the loss.
        loss.backward()

        # Perform single optimization step - parameter update
        optimizer.step()
        
        # Append training loss
        train_loss.append(loss.item())

    # Validation:
    val_loss = []
    val_iou = []
    for image, mask in valid_loader:
        
        image = image.to(device)
        y_pred = model(image)
        
        loss = loss_fn(y_pred, mask.to(device))
        val_loss.append(loss.item())

    print("Epoch: %d, Train: %.3f, Val: %.3f" %
          (e, np.mean(train_loss), np.mean(val_loss)))
val_predictions = []
val_masks = []

for image, mask in tqdm.tqdm(valid_loader):
    image = image.type(torch.float).to(device)
    # Put prediction on CPU, detach it and transform to a numpy array.
    y_pred = model(image).cpu().detach().numpy()
    val_predictions.append(y_pred)
    val_masks.append(mask)


# Stack all masks and predictions along first axis.
# Output of valid_loader is of shape (NxBxCxHxW), where N is number of batches and B is batch size.
val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]


# Cut off padded parts of images.
val_predictions_stacked = val_predictions_stacked[
    :, y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]

val_masks_stacked = val_masks_stacked[
    :, y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]

print(val_masks_stacked.shape, val_predictions_stacked.shape)
random_index = np.random.randint(0, val_masks_stacked.shape[0])
print('Validation Index: {}'.format(random_index))

fig, ax = plt.subplots(2, 1)
ax[0].imshow(val_masks_stacked[random_index], cmap='seismic')
ax[1].imshow(val_predictions_stacked[random_index] > 0.5, cmap='seismic')
test_predictions = []

for image in tqdm.tqdm(test_loader):
    image = image[0].type(torch.float).to(device)
    y_pred = model(image).cpu().detach().numpy()
    test_predictions.append(y_pred)

    
test_predictions_stacked = np.vstack(test_predictions)[:, 0, :, :]
test_predictions_stacked = test_predictions_stacked[:, y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]

print(test_predictions_stacked.shape)
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# To perform RLE, predictions must be in binary integer (0/1) format.
binary_prediction = (test_predictions_stacked > 0.5).astype(int)

# RLE encoding.
all_masks = {idx:rle_encode(binary_prediction[i])
                           for i, idx in enumerate(
                               tqdm.tqdm(test_ids))}
submission = pd.DataFrame.from_dict(all_masks, orient='index')
submission.index.names = ['id']
submission.columns = ['rle_mask']
submission.to_csv('submission.csv')  # 12 epochs score 0.673
