# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import xarray as xr



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



import torch

import torchvision



import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



torch.__version__, torchvision.__version__
ds = xr.open_dataset('/kaggle/input/graphene-kirigami/graphene_processed.nc').astype(np.float32)

ds
cut_density = 1 - ds['coarse_image'].mean(dim=['x_c', 'y_c'])

plt.scatter(ds['strain'], ds['stress'], c=cut_density, cmap=plt.cm.Spectral)

plt.colorbar()
ds['coarse_image'].isel(sample=slice(0, 10)).plot(col='sample', col_wrap=5)
X = ds['coarse_image'].values  # the coarse 3x5 image seems enough

# X = ds['fine_image'].values  # the same model works worse on higher resolution image



y = ds['strain'].values



X = X[..., np.newaxis]  # add channel dimension

y = y[:, np.newaxis]  # pytorch wants ending 1 dimension



# pytorch conv2d wants channel-first, unlike Keras

X = X.transpose([0, 3, 1, 2])  # (sample, x, y, channel) -> (sample, channel, x, y)



X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape
# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader



trainset = torch.utils.data.TensorDataset(

    torch.from_numpy(X_train), torch.from_numpy(y_train)

)



trainloader = torch.utils.data.DataLoader(

    trainset, batch_size=32, shuffle=True, num_workers=2

)
dataiter = iter(trainloader)

inputs, labels = dataiter.next()

inputs.shape, labels.shape  # batch, channel, x, y
class Net(nn.Module):

    def __init__(self,):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        

        # flatten to filter * x * y

        self.fc1 = nn.Linear(64 * 3 * 5, 1)  # coarse grid

        # self.fc1 = nn.Linear(64 * 30 * 80, 1)   # fine grid



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = x.view(-1, 64 * 3 * 5)  # coarse grid

        # x = x.view(-1, 64 * 30 * 80)  # fine grid

        x = self.fc1(x)

        return x



net = Net()
# Check GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
%time net.to(device)
%%time



criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())



for epoch in range(10):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        # inputs, labels = data

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 200 == 199:

            print('[%d, %5d] loss: %.4f' %

                  (epoch + 1, i + 1, running_loss / 200))

            running_loss = 0.0



print('Finished Training')
with torch.no_grad():

    y_train_pred = net(torch.from_numpy(X_train).to(device)).cpu().numpy() # out-of-memory?

    y_test_pred = net(torch.from_numpy(X_test).to(device)).cpu().numpy()



    

# y_train_pred.shape

y_test_pred.shape
r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)
plt.scatter(y_test, y_test_pred, alpha=0.3, s=5)

plt.plot([0, 2], [0, 2], '--', c='k')