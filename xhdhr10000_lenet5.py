# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn.functional as F
class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, is_test=False):

        self.is_test = is_test

        

        data = pd.read_csv(path)

        if not is_test:

            x = torch.tensor(data.values[:,1:]).view(-1,1,28,28)

            self.y = torch.tensor(data.values[:,0])

        else:

            x = torch.tensor(data.values).view(-1,1,28,28)

        self.x = (x.float() - 128) / 128

#         print(f'{self.x.shape} {self.x.dtype} {self.x.mean()} {self.x.min()} {self.x.max()}')



    def __getitem__(self, idx):

        if self.is_test:

            return self.x[idx]

        return self.x[idx], self.y[idx]



    def __len__(self):

        return len(self.x)



# data = Dataset('/kaggle/input/digit-recognizer/test.csv', True)
class Net(torch.nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, 5)

        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.pool = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(16*4*4, 120)

        self.fc2 = torch.nn.Linear(120, 84)

        self.fc3 = torch.nn.Linear(84, 10)

        self.init_weight()



    def init_weight(self):

        torch.nn.init.kaiming_normal_(self.conv1.weight)

        torch.nn.init.kaiming_normal_(self.conv2.weight)

        torch.nn.init.kaiming_normal_(self.fc1.weight)

        torch.nn.init.kaiming_normal_(self.fc2.weight)

        torch.nn.init.kaiming_normal_(self.fc3.weight)

        

    def forward(self, x):

        x = self.pool(self.conv1(x)) # 120x12x12

        x = self.pool(self.conv2(x)) # 80x4x4

        x = x.view(x.size()[0], -1)

        x = self.fc1(x)

        x = self.fc2(x)

        return self.fc3(x)
net = Net()

dataset = Dataset('/kaggle/input/digit-recognizer/train.csv')

train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [round(len(dataset)*0.9), round(len(dataset)*0.1)])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)



criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)



best_acc = 0

for e in range(10):

    epoch_loss = 0

    for s, data in enumerate(train_loader):

        x, y = data

        

        optimizer.zero_grad()

        pred = net(x)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss

        

        if (s+1) % 200 == 0:

            print(f'Step {s+1} loss: {epoch_loss / (s+1):.6}')

    

    # eval

    eval_loss = 0; correct = 0

    for s, data in enumerate(eval_loader):

        x, y = data

        pred = net(x)

        eval_loss += criterion(pred, y)

        correct += torch.nonzero(torch.argmax(pred, axis=1) == y, as_tuple=False).squeeze().size()[0]

    eval_acc = correct / len(eval_dataset)

    print(f'Epoch {e} loss: {epoch_loss / len(train_loader):.6}, eval loss {eval_loss / len(eval_loader):.6} acc {eval_acc:.6}')



    if eval_acc >= best_acc:

        best_acc = eval_acc

        torch.save(net.state_dict(), 'model.pt')
net = Net()

test_dataset = Dataset('/kaggle/input/digit-recognizer/test.csv', True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)



net.load_state_dict(torch.load('model.pt'))



ans = []

for s, data in enumerate(test_loader):

    pred = net(data)

    ans.append(torch.argmax(pred, axis=1).numpy())

ans = np.concatenate(ans)



output = pd.DataFrame({'ImageId': np.arange(1,len(test_dataset)+1), 'Label': ans})

output.to_csv('submit.csv', index=False)