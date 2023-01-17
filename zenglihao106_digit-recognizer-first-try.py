import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import torch 

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
path = '../input/digit-recognizer/'

data_train = pd.read_csv(path+'train.csv')

data_test = pd.read_csv(path+'test.csv')
data_train.columns[:5]
class mnistDataset(Dataset):

    """mnst dataset."""



    def __init__(self, df, train=True):

        self.train = train

        if self.train:

            self.images = np.array(df.drop(['label'],axis=1))

            self.labels = np.array(df.label)

        else:

            self.images = np.array(df)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        if self.train:

            image = self.images[idx].reshape((1,28,28))

            label = self.labels[idx]

            sample = {'image': torch.from_numpy(image), 'label': torch.tensor(label)}

        else:

            image = self.images[idx].reshape((1,28,28))

            sample = {'image': torch.from_numpy(image)}



        return sample
train_ds = mnistDataset(data_train)
def showimg(ds, idx):

    img = ds[idx]['image'].numpy().reshape(28,28)

    print('label: {}'.format(ds[idx]['label']))

    

    plt.imshow(img,cmap='gray')

    plt.show()

    

showimg(train_ds,7)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)

        self.fc1 = nn.Linear(16 * 7 * 7, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 7 * 7)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Assuming that we are on a CUDA machine, this should print a CUDA device:



net.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.0003)
for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_dl):

        inputs = data['image'].float().to(device)

        labels = data['label'].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 1000 == 999:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))

            running_loss = 0.0



print('Finished Training')
correct = 0

total = 0

with torch.no_grad():

    for data in train_dl:

        inputs = data['image'].float().to(device)

        labels = data['label'].to(device)

        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the {} training images: {}'.format(total, round(100 * correct / total,4)))
test_ds = mnistDataset(data_test,train=False)

test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
result = []

with torch.no_grad():

    for data in test_dl:

        inputs = data['image'].float().to(device)

        outputs = net(inputs)

        _, predicted = torch.max(outputs.data,1)

        result.append(predicted)

result = [int(i) for i in result]
img = test_ds[0]['image'].numpy().reshape(28,28)

print('label: {}'.format(result[0]))

plt.imshow(img,cmap='gray')

plt.show()
summit = pd.DataFrame({'ImageId':range(1,len(test_ds)+1),

                      'Label':result})

summit.to_csv('submission.csv',index = False)