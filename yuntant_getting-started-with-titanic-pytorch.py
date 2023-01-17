%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir('/kaggle/input/titanic'))
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df["HasCabin"] = [type(value) is str for value in train_df["Cabin"]]

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']

train_df['IsAlone'] = train_df['FamilySize'] == 0



test_df["HasCabin"] = [type(value) is str for value in test_df["Cabin"]]

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

test_df['IsAlone'] = test_df['FamilySize'] == 0
cat_cols = ["Pclass", "Sex", "HasCabin", 'IsAlone']

num_cols = ['FamilySize']

target_col = 'Survived'
def encode(encoder, x):

    try:

        id = encoder[x]

    except KeyError:

        id = len(encoder)

    return id



encoders = [{} for cat in cat_cols]



for i, cat in enumerate(cat_cols):

    encoders[i] = {l: id for id, l in enumerate(train_df.loc[:, cat].astype(str).unique())}

    train_df[cat] = train_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    test_df[cat] = test_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))



embed_sizes = [len(encoder) for encoder in encoders]
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])

test_df[num_cols] = scaler.fit_transform(test_df[num_cols])
train_df[cat_cols + num_cols]
from sklearn.model_selection import train_test_split



X = train_df[cat_cols + num_cols].values

y = train_df[target_col].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import torchvision

from torchvision import datasets, models, transforms

from torchvision.transforms import functional as F



from tqdm.notebook import trange
def seed_everything(seed=0):

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
net = nn.Sequential(nn.Linear(5, 20), nn.ReLU(), nn.Dropout(0.5), nn.Linear(20, 2))

net
batch_size = 50

num_epochs = 50

learning_rate = 0.01



criterion = nn.CrossEntropyLoss()

# criterion = nn.BCEWithLogitsLoss(reduction="sum")

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



X = torch.tensor(X_train, dtype=torch.float32)

y = torch.tensor(y_train, dtype=torch.long)

train = torch.utils.data.TensorDataset(X, y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)



for epoch in trange(num_epochs):

    net.train()

    # Mini batch learning

    for X_batch, y_batch in train_loader:

        # Forward + Backward + Optimize

        optimizer.zero_grad()

        ypred_var = net(X_batch)

        loss = criterion(ypred_var, y_batch)

        loss.backward()

        optimizer.step()
net.eval()

X = Variable(torch.FloatTensor(X_val), requires_grad=True)

with torch.no_grad():

    result = net(X)

values, labels = torch.max(result, 1)

np.sum(labels.data.numpy() == y_val) / len(y_val)
X_test = torch.tensor(test_df[cat_cols + num_cols].values, dtype=torch.float32)

test_var = Variable(X_test)

result = net(test_var)

values, labels = torch.max(result, 1)



df = pd.DataFrame(columns=['PassengerId', 'Survived'])

df['PassengerId'] = test_df['PassengerId']

df['Survived'] = labels.data.numpy()

df.to_csv('my_submission.csv', index=False)
df