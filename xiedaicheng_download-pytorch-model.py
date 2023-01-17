# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pwd
!pip install requests
import requests


model_urls = {

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    'inception_v3_google':'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',

    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',

    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',

    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',

    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',

    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',

    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',

    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',

    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',

    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',

    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',

    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'#,

    #'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth' #No space left to download this one

}
for (name,url) in model_urls.items():

    filename = url.split('/')[-1]

    print(name)

    r = requests.get(url) 

    with open(filename, "wb") as file:

        file.write(r.content)
!ls
