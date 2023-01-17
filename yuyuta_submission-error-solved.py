# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function, division, absolute_import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch.nn.functional as F

import os



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

# from tqdm import tqdm_notebook as tqdm

from tqdm.notebook import tqdm

import math

import torch.utils.model_zoo as model_zoo



import cv2
class Selayer(nn.Module):



    def __init__(self, inplanes):

        super(Selayer, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)

        self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):



        out = self.global_avgpool(x)



        out = self.conv1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.sigmoid(out)



        return x * out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * 2)



        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,

                               padding=1, groups=cardinality, bias=False)

        self.bn2 = nn.BatchNorm2d(planes * 2)



        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)



        self.selayer = Selayer(planes * 4)



        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



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



        out = self.selayer(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out





class SeResNeXt(nn.Module):

    def __init__(self, block, layers, cardinality=32, num_classes=1000):

        super(SeResNeXt, self).__init__()

        self.cardinality = cardinality

        self.inplanes = 64



        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)



        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:

                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, self.cardinality))

                             

        # vowel_diacritic

        self.fc1 = nn.Linear(2048,11)

        # grapheme_root

        self.fc2 = nn.Linear(2048,168)

        # consonant_diacritic

        self.fc3 = nn.Linear(2048,7)

        return nn.Sequential(*layers)

        



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        

        x1 = self.fc1(x)

        x2 = self.fc2(x)

        x3 = self.fc3(x)

        

        return x1,x2,x3





def se_resnext50(**kwargs):

    """Constructs a SeResNeXt-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = SeResNeXt(Bottleneck, [3, 4, 6, 3],**kwargs)

    return model





def se_resnext101(**kwargs):

    """Constructs a SeResNeXt-101 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = SeResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model





def se_resnext152(**kwargs):

    """Constructs a SeResNeXt-152 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = SeResNeXt(Bottleneck, [3, 8, 36, 3],**kwargs)

    return model
test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
class GraphemeDataset(Dataset):

    def __init__(self,df,_type='train'):

        self.df = df

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        image = self.df.iloc[idx][1:].values.reshape(SIZE,SIZE).astype(float)

        return image, self.df.iloc[idx][0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = se_resnext50().to(device)

# model.load_state_dict(torch.load('/kaggle/input/se-resnext50-baseline/se_resnext50.pth'))

model.load_state_dict(torch.load('/kaggle/input/testnow/try.pth'))
SIZE = 128

def Resize(df,size=SIZE):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
%%time

model.eval()

test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']

predictions1 = []

predictions2 = []

predictions3 = []

row_ids = []

batch_size=256

for fname in test_data:

    start = time.time()

    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    data = Resize(data)

    e_time = time.time() - start

    print ("e_time:{0}".format(e_time) + "[s]")

    test_image = GraphemeDataset(data)

    test_loader = torch.utils.data.DataLoader(test_image,batch_size=batch_size,num_workers=4,shuffle=False)

    with torch.no_grad():

        for inputs,names in tqdm(test_loader):

            for name in names:

                row_ids += [f"{name}_consonant_diacritic",f"{name}_grapheme_root",f"{name}_vowel_diacritic"]

            inputs.to(device)

            

            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float().cuda())

            predictions1.append(outputs3.argmax(1).cpu().detach().numpy())

            predictions2.append(outputs2.argmax(1).cpu().detach().numpy())

            predictions3.append(outputs1.argmax(1).cpu().detach().numpy())
import gc

del model,data,test_image,test_loader

gc.collect()
predictions1 = np.array(predictions1)

predictions1 = predictions1.flatten()

predictions1 = np.hstack(predictions1)

predictions1
predictions2 = np.array(predictions2)

predictions2 = predictions2.flatten()

predictions2 = np.hstack(predictions2)

predictions2
predictions3 = np.array(predictions3)

predictions3 = predictions3.flatten()

predictions3 = np.hstack(predictions3)

predictions3
pred = [[predictions1[i],predictions2[i],predictions3[i]] for i in range(len(predictions1))]

pred = np.hstack(np.hstack(pred))

pred
# # submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

# submission['row_id'] = row_ids

# submission['target'] = pred

submission = pd.DataFrame({'row_id':row_ids,'target':pred},columns=['row_id','target'])

submission
submission.to_csv('submission.csv',index=False)