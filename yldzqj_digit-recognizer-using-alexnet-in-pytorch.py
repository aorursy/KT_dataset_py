import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import torch

import torch.nn as nn

from torchvision import transforms, models

from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim

import torch.functional as F

from sklearn.model_selection import train_test_split
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
x_train, x_val, y_train, y_val = train_test_split(train_data.values[:, 1:], train_data.values[:, 0], test_size=0.2) 
batch_size = 64

num_epochs = 50
x_train_tensor = torch.from_numpy(x_train.astype(np.float32)/255).view(-1, 1, 28, 28)

y_train_tensor = torch.from_numpy(y_train)

x_val_tensor = torch.from_numpy(x_val.astype(np.float32)/255).view(-1, 1, 28, 28)

y_val_tensor = torch.from_numpy(y_val)

test_tensor = torch.from_numpy(test_data.values[:,:].astype(np.float32)/255).view(-1, 1, 28, 28)



train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

test_dataset = TensorDataset(test_tensor)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
class Net(nn.Module):

    def __init__(self, num_classes=10):

        super(Net, self).__init__()

        self.feature = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size = 3, stride = 2, padding = 1, bias = False), # 14 * 14 * 96

            nn.BatchNorm2d(96),

            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 7 * 7 * 96

            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1,bias=False),  # 7*7*256

            nn.BatchNorm2d(256),

            nn.ReLU(True),

            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=False),     # 4*4*384

            nn.BatchNorm2d(384),

            nn.ReLU(True),

            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=False),    # 2*2*384

            nn.BatchNorm2d(384),

            nn.ReLU(True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*384

            nn.BatchNorm2d(384),

            nn.ReLU(True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*256

            nn.BatchNorm2d(256),

            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2)                       # 1*1*256

        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):

        out = self.feature(x)

        out = out.view(out.size(0), -1)

        out = self.classifier(out)

        return out
net = Net().cuda()

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

criterion = nn.CrossEntropyLoss()
def test():

    net.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for i, data in enumerate(val_loader):

            images, labels = data

            images = images.cuda()

            labels = labels.cuda()

            outputs_test = net(images)

            _, predicted = outputs_test.max(1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

        acc = correct / total

        return acc
best_acc = 0.0

for epoch in range(num_epochs):

    net.train()

    sum_loss = 0.0

    for i, data in enumerate(train_loader):

        inputs, labels = data

        inputs = inputs.cuda()

        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        sum_loss += loss.item()

        if i % 100 == 99:

            print('[%d %d] loss:%.03f' % (epoch+1, i+1, sum_loss / 100))

            sum_loss = 0.0

acc = test()

print(acc)
net.eval()

preds = []

for i, data in enumerate(test_loader):

    image = data[0]

    image = image.cuda()

    output = net(image)

    _, prediction = output.max(1)

    preds.extend(prediction.tolist())
sample_submission['Label'] = preds

sample_submission.to_csv('submission.csv', index = False)