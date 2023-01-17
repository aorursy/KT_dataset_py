!pip install -U skorch
import numpy as np

import pandas as pd

import os

import cv2

import torch

import torchvision

import torch.nn as nn

import torch.optim as optim

from PIL import Image

from torchvision import models, transforms

from torch.utils.data import Dataset, DataLoader

from skorch import NeuralNetClassifier

from sklearn.model_selection import cross_val_score
ROOT_DIR = '../input/images'

train_df = pd.read_csv('../input/train.csv')

pred_df = pd.read_csv('../input/sample_submission.csv')

CLASSES = pred_df.columns[1:].tolist()

NUM_CLASSES = len(CLASSES)

torch.set_default_tensor_type('torch.DoubleTensor')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(65536, 1000)

        self.fc2 = nn.Linear(1000, NUM_CLASSES)    

    

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)

        out = self.fc1(out)

        out = self.fc2(out)

        return out
X_train = []

y_train = []

for index, row in train_df.iterrows():

    path = os.path.join(ROOT_DIR, str(row['id']) + '.jpg')

    img = cv2.imread(path)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (224, 224))

    img = np.transpose(img, (2,1,0))

    #mg = np.expand_dims(img, axis=0)

    img = img / 255.0

    X_train.append(img)

    y_train.append(CLASSES.index(row['species']))
X_train = np.array(X_train)

y_train = np.array(y_train)

print (X_train.shape, y_train.shape)
#model = ConvNet()

#model.to(DEVICE)

model = models.resnet34(pretrained=True)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model.to(DEVICE)
classifier = NeuralNetClassifier(

            module = model, 

            max_epochs = 20, 

            lr = 1e-4,

            criterion = nn.CrossEntropyLoss,

            optimizer=optim.Adam,

            device=DEVICE)
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv = 5, scoring = "accuracy")
print(scores)

print(scores.mean(), scores.std())
X_test = []

test_df = pd.read_csv("../input/test.csv")

for index, row in test_df.iterrows():

    path = os.path.join(ROOT_DIR, str(int(row['id'])) + '.jpg')

    img = cv2.imread(path)

    img = cv2.resize(img, (224, 224))

    img = np.transpose(img, (2,1,0))

    img = img / 255.0

    X_test.append(img)
preds = classifier.predict_proba(np.array(X_test))

preds = torch.nn.functional.softmax(torch.tensor(preds), dim=1).numpy()

print (preds)
pred_df = pd.read_csv('../input/sample_submission.csv')

for index, row in pred_df.iterrows():

    pred_df.loc[pred_df['id'] == row['id'],1:] = preds[index]
pred_df
pred_df.to_csv('../working/prediction.csv', index=False)