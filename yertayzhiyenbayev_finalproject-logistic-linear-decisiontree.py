import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import numpy as np

import os
# for dirname, _, filenames in os.walk('/kaggle/input/state-farm-distracted-driver-detection/imgs'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import os, glob, math, cv2, time
import numpy as np
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

img_size = 128
def process_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(img_size,img_size))
    return img
start = time.time()

X_data = []
Y_data = []

for j in range(10):
    print('Load folder c{}'.format(j))
    path = os.path.join('../input/state-farm-distracted-driver-detection/imgs/train', 'c' + str(j), '*.jpg')
    files = glob.glob(path)
    for fl in files:
            flbase = os.path.basename(fl)
            img = process_image(fl)
            X_data.append(img)
            Y_data.append(j)
    
end = time.time() - start
print("Time: %.2f seconds" % end)
X_data = np.array(X_data)
Y_data = np.array(Y_data)
plt.imshow(X_data[0],cmap= 'gray')
plt.show()
print(Y_data[0])
X_data = np.reshape(X_data, (X_data.shape[0], -1))
Y_data = np.reshape(Y_data, (-1, 1))
class Drivers_dataset(Dataset):
    def __init__(self, df):
        rows = df.shape[0]
        self.imgnp = df.iloc[:rows, 0:img_size*img_size].values
        self.labels = df.iloc[:rows, img_size*img_size].values
        self.rows = rows
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        image = torch.tensor(self.imgnp[idx], dtype=torch.float) / 255  # Normalize
        image = image.view(1, img_size, img_size)  # (channel, height, width)
        label = self.labels[idx]
        return (image, label)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state = 0, test_size = 1/5 )
del X_data, Y_data
trainset = np.append(X_train, np.reshape(Y_train, (-1, 1)), axis = 1)
testset = np.append(X_test, np.reshape(Y_test, (-1, 1)), axis = 1)
testset = pd.DataFrame(data = testset)
trainset = pd.DataFrame(data = trainset)
trainset = Drivers_dataset(trainset)
testset = Drivers_dataset(testset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                          shuffle=True, num_workers=2)
train_data_iter = iter(trainloader)
test_data_iter = iter(testloader)
dataiter = iter(trainloader)
images, labels = dataiter.next()

plt.imshow(images.numpy()[0,0,::],cmap= 'gray')
plt.show()

images.size(), labels.size()
for data in trainloader:
  inputs, labels = data
  print(inputs.shape)
  print(labels.shape)
  print(labels.data)
  break
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

# Parameters:
# in_channels (int) – Number of channels in the input image
# out_channels (int) – Number of channels produced by the convolution
# kernel_size (int or tuple) – Size of the convolving kernel (Filter size)
# stride (int or tuple, optional) – Stride of the convolution. (Default: 1)
# padding (int or tuple, optional) – Zero-padding added to both sides of the input (Default: 0)
# padding_mode (string, optional) – zeros
# dilation (int or tuple, optional) – Spacing between kernel elements. (Default: 1)
# groups (int, optional) – Number of blocked connections from input to output channels. (Default: 1)
# bias (bool, optional) – If True, adds a learnable bias to the output. (Default: True)
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        # Define hidden convolutional layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define hidden linear layers
        self.fc1 = nn.Linear(128 * 14 * 14, 120)
#         self.fc1_drop = nn.Dropout(p = 0.1)
        self.fc2 = nn.Linear(120, 84)
#         self.fc2_drop = nn.Dropout(p = 0.1)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 14 * 14)
        
        x = F.relu(self.fc1(x))
#         x = self.fc1_drop(x)

        x = F.softmax(self.fc2(x), dim = 1)
#         x = self.fc2_drop(x)

        x = self.fc3(x)
        
        return x
net = Net()
import torch.optim as optim
import torch.backends.cudnn as cudnn

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

if torch.cuda.is_available():
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
num_epoch = 33

for epoch in range(num_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # inputs, labels = Variable(inputs), Variable(labels)
        
        labels = labels.type(torch.LongTensor)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs.cuda(), labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')
correct = 0
total = 0

# for data in testloader:
for i, data in enumerate(testloader, 0):
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the {} test images: {:4.2f} %'.format(
    X_test.shape[0], 100 * correct / total))
