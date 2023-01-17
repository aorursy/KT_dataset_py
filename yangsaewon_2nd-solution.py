import gc

import os

import time

import random

import numpy as np

import pandas as pd

from pathlib import Path

import glob

import PIL

from PIL import Image, ImageEnhance, ImageOps

from collections import OrderedDict

from joblib import Parallel, delayed



from tqdm import tqdm, tqdm_notebook



import torch

from torch import nn, cuda

import torch.nn.functional as F

import torchvision.transforms as transforms

import torchvision.models as M

from torch.utils.data import Dataset, DataLoader
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 2019

seed_everything(SEED)
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
class TestDataset(Dataset):

    def __init__(self, test_imgs, transforms=None):

        self.test_imgs = test_imgs

        self.transform = transforms

        

    def __len__(self):

        return len(self.test_imgs)

    

    def __getitem__(self, idx):

        

        image = self.test_imgs[idx].convert('RGB')

            

        if self.transform:

            image = self.transform(image)

            

        return image        
df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/train.csv')

DATA_PATH = '../input/2019-3rd-ml-month-with-kakr'

TEST_IMAGE_PATH = os.path.join(DATA_PATH, 'test')

test_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/test.csv')

num_classes = df['class'].nunique()
%%time



x_test = Parallel(n_jobs=4)(

    delayed(lambda x: crop_boxing_img(x))(x) for x in tqdm(test_df['img_file']))
# target_size = (224, 224)



# data_transforms = transforms.Compose([

#         transforms.Resize(target_size),

#         transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),

#         transforms.RandomHorizontalFlip(),

#         transforms.ToTensor(),

#         transforms.Normalize(

#             [0.485, 0.456, 0.406], 

#             [0.229, 0.224, 0.225])

#     ])
# resnet50_weights_path = Path('../input/car-4fold-weights/')

# weight_list = os.listdir(resnet50_weights_path)

# weight_list
# %%time



# batch_size = 1

# tta = 5

# test_dataset = TestDataset(x_test, transforms=data_transforms)

# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# total_num_models = len(weight_list)*tta 



# model = M.resnet50() 

# model.fc = nn.Linear(2048, num_classes)

# model.cuda()



# resnet50_prediction = np.zeros((len(test_dataset), num_classes))



# for i, weight in enumerate(weight_list):

#     print("weight {} prediction starts".format(i+1))

    

#     for _ in range(tta):

#         print("tta {}".format(_+1))



#         model.load_state_dict(torch.load(resnet50_weights_path / weight))



#         model.eval()

        

#         prediction = np.zeros((len(test_dataset), num_classes))

#         with torch.no_grad():

#             for i, images in enumerate(test_loader):

#                 images = images.cuda()



#                 preds = model(images).detach()

#                 prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

#                 resnet50_prediction = resnet50_prediction + prediction

#         del prediction

        

# resnet50_prediction /= total_num_models



# del test_dataset

# del test_loader

# gc.collect()
class Bottleneck(nn.Module):

    """

    Base class for bottlenecks that implements `forward()` method.

    """

    def forward(self, x):

        residual = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out = self.se_module(out) + residual

        out = self.relu(out)



        return out



class SEBottleneck(Bottleneck):

    """

    Bottleneck for SENet154.

    """

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None):

        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,

                               stride=stride, padding=1, groups=groups,

                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes * 4)

        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,

                               bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride

        

        

class SEModule(nn.Module):



    def __init__(self, channels, reduction):

        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,

                             padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,

                             padding=0)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        module_input = x

        x = self.avg_pool(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return module_input * x



class SENet(nn.Module):



    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,

                 inplanes=128, input_3x3=True, downsample_kernel_size=3,

                 downsample_padding=1, num_classes=1000):

        """

        Parameters

        ----------

        block (nn.Module): Bottleneck class.

            - For SENet154: SEBottleneck

            - For SE-ResNet models: SEResNetBottleneck

            - For SE-ResNeXt models:  SEResNeXtBottleneck

        layers (list of ints): Number of residual blocks for 4 layers of the

            network (layer1...layer4).

        groups (int): Number of groups for the 3x3 convolution in each

            bottleneck block.

            - For SENet154: 64

            - For SE-ResNet models: 1

            - For SE-ResNeXt models:  32

        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.

            - For all models: 16

        dropout_p (float or None): Drop probability for the Dropout layer.

            If `None` the Dropout layer is not used.

            - For SENet154: 0.2

            - For SE-ResNet models: None

            - For SE-ResNeXt models: None

        inplanes (int):  Number of input channels for layer1.

            - For SENet154: 128

            - For SE-ResNet models: 64

            - For SE-ResNeXt models: 64

        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of

            a single 7x7 convolution in layer0.

            - For SENet154: True

            - For SE-ResNet models: False

            - For SE-ResNeXt models: False

        downsample_kernel_size (int): Kernel size for downsampling convolutions

            in layer2, layer3 and layer4.

            - For SENet154: 3

            - For SE-ResNet models: 1

            - For SE-ResNeXt models: 1

        downsample_padding (int): Padding for downsampling convolutions in

            layer2, layer3 and layer4.

            - For SENet154: 1

            - For SE-ResNet models: 0

            - For SE-ResNeXt models: 0

        num_classes (int): Number of outputs in `last_linear` layer.

            - For all models: 1000

        """

        super(SENet, self).__init__()

        self.inplanes = inplanes

        if input_3x3:

            layer0_modules = [

                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,

                                    bias=False)),

                ('bn1', nn.BatchNorm2d(64)),

                ('relu1', nn.ReLU(inplace=True)),

                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,

                                    bias=False)),

                ('bn2', nn.BatchNorm2d(64)),

                ('relu2', nn.ReLU(inplace=True)),

                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,

                                    bias=False)),

                ('bn3', nn.BatchNorm2d(inplanes)),

                ('relu3', nn.ReLU(inplace=True)),

            ]

        else:

            layer0_modules = [

                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,

                                    padding=3, bias=False)),

                ('bn1', nn.BatchNorm2d(inplanes)),

                ('relu1', nn.ReLU(inplace=True)),

            ]

        # To preserve compatibility with Caffe weights `ceil_mode=True`

        # is used instead of `padding=1`.

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,

                                                    ceil_mode=True)))

        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self._make_layer(

            block,

            planes=64,

            blocks=layers[0],

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=1,

            downsample_padding=0

        )

        self.layer2 = self._make_layer(

            block,

            planes=128,

            blocks=layers[1],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.layer3 = self._make_layer(

            block,

            planes=256,

            blocks=layers[2],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.layer4 = self._make_layer(

            block,

            planes=512,

            blocks=layers[3],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        self.last_linear = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,

                    downsample_kernel_size=1, downsample_padding=0):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=downsample_kernel_size, stride=stride,

                          padding=downsample_padding, bias=False),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, groups, reduction, stride,

                            downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups, reduction))



        return nn.Sequential(*layers)



    def features(self, x):

        x = self.layer0(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x



    def logits(self, x):

        x = self.avg_pool(x)

        if self.dropout is not None:

            x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.last_linear(x)

        return x



    def forward(self, x):

        x = self.features(x)

        x = self.logits(x)

        return x





def senet154(num_classes=1000, pretrained='imagenet'):

    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,

                  dropout_p=0.2, num_classes=num_classes)

    if pretrained is not None:

        settings = pretrained_settings['senet154'][pretrained]

        initialize_pretrained_model(model, num_classes, settings)

    return model
target_size = (224, 224)



data_transforms = transforms.Compose([

            transforms.Resize(target_size),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(

                [0.485, 0.456, 0.406], 

                [0.229, 0.224, 0.225])

])
senet154_weights_path = Path('../input/car-senet142/')

weight_list = os.listdir(senet154_weights_path)

weight_list
%%time



batch_size = 1

tta = 2

test_dataset = TestDataset(x_test, transforms=data_transforms)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

total_num_models = len(weight_list)*tta 



model = senet154(num_classes=1000, pretrained=None)

model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)

model.cuda()



senet154_prediction = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("weight {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        model.load_state_dict(torch.load(senet154_weights_path / weight))



        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes))

        with torch.no_grad():

            for i, images in enumerate(test_loader):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                senet154_prediction = senet154_prediction + prediction

        del prediction

    

    

senet154_prediction /= total_num_models
resnext_weights_path = Path('../input/car-resnext-weights/')

weight_list = os.listdir(resnext_weights_path)

weight_list
%%time



batch_size = 1

tta = 4

total_num_models = len(weight_list)*tta 



model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, num_classes))

model.cuda()



resnext_prediction = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("weight {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        model.load_state_dict(torch.load(resnext_weights_path / weight))



        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes))

        with torch.no_grad():

            for i, images in enumerate(test_loader):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                resnext_prediction = resnext_prediction + prediction

        del prediction

        

    

resnext_prediction /= total_num_models
all_prediction = (senet154_prediction/2) + (resnext_prediction/2)

# all_prediction = (resnet50_prediction/3) + (resnext_prediction/3) + (senet154_prediction/3)



result = np.argmax(all_prediction, axis=1)

submission = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/sample_submission.csv')

submission["class"] = result

submission["class"].replace(0, 196, inplace=True)

submission.to_csv("submission.csv", index=False)

submission.head()