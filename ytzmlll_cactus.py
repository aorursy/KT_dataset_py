# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pd.read_csv('../input/train.csv')

import matplotlib.pyplot as plt

%matplotlib inline  

from PIL import Image

all_images_fnames = os.listdir("../input/train/train")

for i in all_images_fnames[:10]:

    image = Image.open("../input/train/train/{}".format(i))

    image_numpy = np.asarray(image)

    from matplotlib.pyplot import imshow

    imshow(np.asarray(image_numpy))

    plt.show()
import torchvision

import PIL

#https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=9NRlYXKQy3Kx

transformations_rotation = torchvision.transforms.Compose([

    torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, hue=.2, saturation=.2),

    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.RandomVerticalFlip(),

    torchvision.transforms.RandomRotation(degrees = (90,90))

])

transformations_no_rotation = torchvision.transforms.Compose([

    torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, hue=.2, saturation=.2),

    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.RandomVerticalFlip()

])
train_labels_df = pd.read_csv('../input/train.csv')

print("How many pictures contain cactuses?")

train_labels_df['has_cactus'].sum() / train_labels_df['has_cactus'].count()
all_pictures_data = [Image.open("../input/train/train/{}".format(i)) for i in list(train_labels_df['id'])]
augumented_pictures_data =  []

augumented_pictures_labels = []

for picture, label in zip(all_pictures_data, train_labels_df['has_cactus'].values):

    augumented_pictures_data.append(picture)

    augumented_pictures_labels.append(label)

    for i in range(4):

        augumented_pictures_data.append(transformations_rotation(picture))

        augumented_pictures_data.append(transformations_no_rotation(picture))

        augumented_pictures_labels.append(label)

        augumented_pictures_labels.append(label)

for pic in augumented_pictures_data[:8]:

    from matplotlib.pyplot import imshow

    imshow(np.asarray(pic))

    plt.show()
X = torch.Tensor(np.asarray([np.asarray(i) for i in augumented_pictures_data])).cuda()

print(X.shape)

X = X.reshape(17500*9, 3, 32, 32)
labels = np.asarray(augumented_pictures_labels)

y = torch.Tensor(labels).cuda()

print(y)
from torchvision.models import vgg16

model = vgg16(pretrained=True, progress=True)

model.cuda()

model.classifier[6]
num_features = model.classifier[6].in_features

features = list(model.classifier.children())[:-1] # Remove last layer

features.extend([torch.nn.Linear(num_features, 2)]) # Add our layer with 4 outputs

model.classifier = torch.nn.Sequential(*features) # Replace the model classifier



for param in model.features.parameters():

    param.require_grad = False

model.cuda()

print(model)
from  torch.utils import data

import random

X.shape

n = X.shape[0]

train_indexes_count = int(n*0.1)

test_indexes_count = n - train_indexes_count





from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader



dataset = TensorDataset(X, y)

train_set, test_set = data.random_split(dataset, (train_indexes_count, test_indexes_count))

loader = DataLoader(train_set, batch_size = 128)

for epoch in range(30):

    for X_sample, y_sample in loader:

        print(y_sample)

        break





learning_rate = 0.0001

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(60):

    losses = []

    for batch_x, batch_y in loader:

        y_pred = model(batch_x)

        #print(labels_tensor)

        #print(y_pred.argmax(dim=1))

        loss = loss_fn(y_pred, batch_y.long())

        losses.append(loss.item())

        model.zero_grad()

        loss.backward()

        optimizer.step()

    print(epoch, sum(losses) / len(losses))
from sklearn.metrics import classification_report



loader = DataLoader(test_set, batch_size = 128)

y_pred = []

for batch in loader:

    y_pred += [i for i in model(batch[0]).argmax(1).cpu().numpy()]

y_test = [int(i[1].cpu().numpy()) for i in test_set]
print(classification_report(y_test, y_pred))
y_baseline = [1 for i in y_test]

print(classification_report(y_baseline, y_pred))
test_pictures_data = [np.asarray(Image.open("../input/test/test/{}".format(i))) for i in os.listdir("../input/test/test")]

len(test_pictures_data)
X_submit = torch.Tensor(np.asarray(test_pictures_data)).cuda()

X_submit = X_submit.reshape(4000, 3, 32, 32)

#print(X_submit.shape)

y_submit = model(X_submit)
y_submit_list = list(y_submit.argmax(1).cpu().numpy())

with open('to_submit.csv', 'w') as fhout:

    fhout.write('id,has_cactus\n')

    for fname, y in zip(os.listdir("../input/test/test"), y_submit_list):

        fhout.write("{}, {}\n".format(fname, y))