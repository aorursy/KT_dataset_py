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
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_train = train_df.iloc[:, 1:].values.reshape(-1, 1, 28, 28) / 255

y_train = train_df.iloc[:, 0].values
plt.imshow(X_train[3][0], cmap='gray')
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
import torch

import torch.nn as nn

from torch.nn.functional import relu
def seed_everything(seed=0):

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv1 = nn.Conv2d(1, 16, 3) # 畳み込み層．(入力のチャンネル数、畳み込み後のチャンネル数、正方形フィルタの一辺)、

        self.pool = nn.MaxPool2d(2, 2) # プーリング層．データサイズを縮小する．

        self.conv2 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 5 * 5, 120) # 全結合層

        self.fc2 = nn.Linear(120, 10)

        

    def forward(self, x):

        x = self.pool(relu(self.conv1(x))) # 画像サイズは28pxから13pxになる

        x = self.pool(relu(self.conv2(x))) # 画像サイズは5pxになる

        x = x.view(x.size()[0], 32 * 5 * 5) # 全結合層に渡すためにテンソルを変形する

        x = relu(self.fc1(x))

        x = self.fc2(x)

        return x

    

net = Net()

net
import torch.optim as optim

from tqdm.notebook import trange



batch_size = 100

num_epochs = 100

learning_rate = 0.001

momentum = 0.9



device = torch.device('cuda')

net = net.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)



X = torch.FloatTensor(X_train)

y = torch.LongTensor(y_train)

train = torch.utils.data.TensorDataset(X, y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)



for epoch in trange(num_epochs):

    net.train()

    # Mini batch learning

    for X_batch, y_batch in train_loader:

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward + Backward + Optimize

        optimizer.zero_grad()

        ypred_var = net(X_batch)

        loss = criterion(ypred_var, y_batch)

        loss.backward()

        optimizer.step()
net.eval()

X = torch.FloatTensor(X_val).to(device)

output = net(X)

_, y_pred = torch.max(output, 1)
np.sum(y_pred.cpu().numpy() == y_val) / len(y_val)
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_test = test_df.values.reshape(-1, 1, 28, 28) / 255
X = torch.FloatTensor(X_test).to(device)

output = net(X)

_, y_pred = torch.max(output, 1)
df = pd.DataFrame(columns=['ImageId', 'Label'])

df['ImageId'] = np.arange(1, len(test_df) + 1)

df['Label'] = y_pred.cpu().numpy()

df.to_csv('my_submission.csv', index=False)