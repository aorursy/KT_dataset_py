import gc

import os

import time

import random

import numpy as np

import pandas as pd

from pathlib import Path

import PIL

import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance, ImageOps



from tqdm import tqdm, tqdm_notebook



import torch

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
'''

AutoAugment: Learning Augmentation Policies from Data

https://arxiv.org/abs/1905.

'''



class ImageNetPolicy(object):

    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:

        >>> policy = ImageNetPolicy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     ImageNetPolicy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),

            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),



            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),

            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),

            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),

            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),

            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),



            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),

            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),

            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),

            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),

            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),



            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),

            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),

            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),

            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),

            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),



            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),

            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),

            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment ImageNet Policy"



class SVHNPolicy(object):

    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:

        >>> policy = SVHNPolicy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     SVHNPolicy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),

            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),

            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),

            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),



            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),

            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),

            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),

            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),

            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),



            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),

            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),

            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),

            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),

            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),



            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),

            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),

            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),

            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),



            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),

            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),

            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),

            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),

            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment SVHN Policy"





class CIFAR10Policy(object):

    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:

        >>> policy = CIFAR10Policy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     CIFAR10Policy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),

            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),

            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),

            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),

            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),



            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),

            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),

            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),

            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),

            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),



            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),

            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),

            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),

            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),

            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),



            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),

            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),

            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),

            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),

            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),



            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),

            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),

            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),

            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment CIFAR10 Policy"



    

class SubPolicy(object):

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):

        ranges = {

            "shearX": np.linspace(0, 0.3, 10),

            "shearY": np.linspace(0, 0.3, 10),

            "translateX": np.linspace(0, 150 / 331, 10),

            "translateY": np.linspace(0, 150 / 331, 10),

            "rotate": np.linspace(0, 30, 10),

            "color": np.linspace(0.0, 0.9, 10),

            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),

            "solarize": np.linspace(256, 0, 10),

            "contrast": np.linspace(0.0, 0.9, 10),

            "sharpness": np.linspace(0.0, 0.9, 10),

            "brightness": np.linspace(0.0, 0.9, 10),

            "autocontrast": [0] * 10,

            "equalize": [0] * 10,

            "invert": [0] * 10

        }



        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand

        def rotate_with_fill(img, magnitude):

            rot = img.convert("RGBA").rotate(magnitude)

            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)



        func = {

            "shearX": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),

                Image.BICUBIC, fillcolor=fillcolor),

            "shearY": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),

                Image.BICUBIC, fillcolor=fillcolor),

            "translateX": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),

                fillcolor=fillcolor),

            "translateY": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),

                fillcolor=fillcolor),

            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),

            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),

            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),

            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),

            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),

            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),

            "equalize": lambda img, magnitude: ImageOps.equalize(img),

            "invert": lambda img, magnitude: ImageOps.invert(img)

        }



        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(

        #     operation1, ranges[operation1][magnitude_idx1],

        #     operation2, ranges[operation2][magnitude_idx2])

        self.p1 = p1

        self.operation1 = func[operation1]

        self.magnitude1 = ranges[operation1][magnitude_idx1]

        self.p2 = p2

        self.operation2 = func[operation2]

        self.magnitude2 = ranges[operation2][magnitude_idx2]





    def __call__(self, img):

        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)

        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)

        return img
'''

허태명님 kernel

https://www.kaggle.com/tmheo74/3rd-ml-month-car-image-cropping

'''

def crop_boxing_img(img_name, margin=16) :

    if img_name.split('_')[0] == "train" :

        PATH = TRAIN_IMAGE_PATH

        data = train_df

    elif img_name.split('_')[0] == "test" :

        PATH = TEST_IMAGE_PATH

        data = test_df

        

    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                   ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)



    return img.crop((x1,y1,x2,y2))







def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
class TestDataset(Dataset):

    def __init__(self, df, mode='original', transforms=None, crop=False):

        self.df = df

        self.mode = mode

        self.transforms = transforms[self.mode]

        self.crop = crop

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        if self.crop:

            image = crop_boxing_img(self.df['img_file'][idx]).convert('RGB')

        else:

            image = Image.open(os.path.join(TEST_IMAGE_PATH, self.df['img_file'][idx])).convert("RGB")

            

        if self.transforms:

            image = self.transforms(image)

            

        return image        



    

target_size = (224, 224)





data_transforms = {

    'original': transforms.Compose([

        transforms.Resize(target_size),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'augment': transforms.Compose([

        transforms.Resize(target_size),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(20),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'cifar10_autoaug': transforms.Compose([

        transforms.Resize(target_size),

        CIFAR10Policy(),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'SVHN_autoaug': transforms.Compose([

        transforms.Resize(target_size),

        SVHNPolicy(),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'ImageNet_autoaug': transforms.Compose([

        transforms.Resize(target_size),

        ImageNetPolicy(),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'resized_crop': transforms.Compose([

        transforms.RandomResizedCrop(target_size),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ])

}

DATA_PATH = '../input/'

TEST_IMAGE_PATH = os.path.join(DATA_PATH, 'test')

test_df = pd.read_csv('../input/test.csv')
batch_size = 4

test_dataset = TestDataset(test_df, mode='original', transforms=data_transforms, crop=False)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)
test_dataset = TestDataset(test_df, mode='original', transforms=data_transforms, crop=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)
test_dataset = TestDataset(test_df, mode='augment', transforms=data_transforms, crop=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)
test_dataset = TestDataset(test_df, mode='cifar10_autoaug', transforms=data_transforms, crop=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)
test_dataset = TestDataset(test_df, mode='SVHN_autoaug', transforms=data_transforms, crop=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)
test_dataset = TestDataset(test_df, mode='ImageNet_autoaug', transforms=data_transforms, crop=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)
test_dataset = TestDataset(test_df, mode='resized_crop', transforms=data_transforms, crop=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



inputs = next(iter(test_loader))

out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=(30, 30))

imshow(out)