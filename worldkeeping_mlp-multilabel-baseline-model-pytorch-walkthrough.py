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

from torch import nn

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import gc

import sys

from sklearn.metrics import log_loss

from scipy.special import expit

import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

trainraw = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
featurename = trainraw.columns.tolist()

genefeat = [n for n in featurename if n[:2]=='g-']

cellfeat = [n for n in featurename if n[:2]=='c-']
f, ax = plt.subplots(1, 2, figsize = (15,3))

ax[0].plot(trainraw.loc[0,genefeat].values)

ax[0].set_title('gene expression')

ax[1].plot(trainraw.loc[0,cellfeat].values)

ax[1].set_title('cell viability')

plt.show()
OHE = OneHotEncoder(sparse=False)

onehotfeat = OHE.fit_transform(trainraw[['cp_type', 'cp_dose']])

onehotfeat.shape
trainraw['cp_type_0'] = onehotfeat[:,0]

trainraw['cp_type_1'] = onehotfeat[:,1]

trainraw['cp_dose_0'] = onehotfeat[:,2]

trainraw['cp_dose_1'] = onehotfeat[:,3]

trainraw.drop(['cp_type', 'cp_dose'], axis=1, inplace=True)
test_type = test['cp_type']

onehotfeat = OHE.transform(test[['cp_type', 'cp_dose']])

onehotfeat.shape

test['cp_type_0'] = onehotfeat[:,0]

test['cp_type_1'] = onehotfeat[:,1]

test['cp_dose_0'] = onehotfeat[:,2]

test['cp_dose_1'] = onehotfeat[:,3]

test.drop(['cp_type', 'sig_id', 'cp_dose'], axis=1, inplace=True)
wholeset = targets.merge(trainraw, how='left', on='sig_id')

wholeset.drop('sig_id', axis=1, inplace=True)

targetsname = targets.columns.tolist()

targetsname.remove('sig_id')

wholeset = wholeset[wholeset['cp_type_0']!=1].reset_index(drop=True)

targets = wholeset[targetsname]

wholeset.drop(targetsname, axis=1, inplace=True)
plt.figure(figsize=(20,5))

plt.xticks(rotation=90)

plt.bar(targetsname,targets.sum(axis=0))
#split data 

# train_x, valid_x, train_y, valid_y = train_test_split(wholeset, targets, test_size=0.2, shuffle=True, random_state=111914)
class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, labels):

        'Initialization'

        self.labels = labels

        self.X = X

        

    def __len__(self):

        'Denotes the total number of samples'

        return len(self.X)



    def __getitem__(self, index):

        'Generates one sample of data'

        # Load data and get label

        x = self.X[index]

        y = self.labels[index]

        return x, y
class Mynet(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(877, 1024),

#             nn.BatchNorm1d(1024),

            nn.ReLU(),

            nn.Linear(1024, 512),

#             nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 206),

            )  

    def forward(self, x):

        x = self.mlp(x)

        return x
#capture gpu if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# train_x, train_y, valid_x, valid_y = train_x.values, train_y.values, valid_x.values, valid_y.values
# #load data

# total_size = train_x.shape[0]

# batch_size = 256

# batchs = int(np.ceil(total_size/batch_size))

# valid_loader = torch.utils.data.DataLoader(valid_x.astype(np.float32), batch_size=32, shuffle=False)

# train_loader = torch.utils.data.DataLoader(train_x.astype(np.float32), batch_size=32, shuffle=False)

# trainlabeleddata = Dataset(train_x.astype(np.float32),train_y.astype(np.float32))

# trainlabeleddata_loader = torch.utils.data.DataLoader(trainlabeleddata, batch_size=batch_size, shuffle=True)
def log_loss_metric(y_true, y_pred):

    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = - np.mean(np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip), axis = 1))

    return loss
nfolds = 10

Kfolds = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True, random_state=234)

test_loader = torch.utils.data.DataLoader(test.values.astype(np.float32), batch_size=32, shuffle=False)
oof = np.zeros(targets.shape)

predictions = np.zeros((test.shape[0], targets.shape[1]))

for f, (t_idx, v_idx) in enumerate(Kfolds.split(X=wholeset, y=targets)):

    print(f'fold_{f+1}', flush=True)

    #split data

    train_x =  wholeset.values[t_idx,:]

    train_y =  targets.values[t_idx,:] 

    valid_x =  wholeset.values[v_idx,:]

    valid_y =  targets.values[v_idx,:]



    #load data

    total_size = train_x.shape[0]

    batch_size = 256

    batchs = int(np.ceil(total_size/batch_size))

    valid_loader = torch.utils.data.DataLoader(valid_x.astype(np.float32), batch_size=32, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_x.astype(np.float32), batch_size=32, shuffle=False)

    trainlabeleddata = Dataset(train_x.astype(np.float32),train_y.astype(np.float32))

    trainlabeleddata_loader = torch.utils.data.DataLoader(trainlabeleddata, batch_size=batch_size, shuffle=True)



    #initialize model

    mynet = Mynet()

    mynet.to(device)

    lr = 0.001

    num_epochs = 50

    loss_function = nn.BCEWithLogitsLoss()



    optimizer = torch.optim.Adamax(mynet.parameters(), lr=lr)

#     lambdaschecule = lambda x: 0.95

#     scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambdaschecule)

#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 

#                                               max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(trainlabeleddata_loader))

    losses = []

    vallosses = []

    acc = []

    valacc = []

    gc.collect()

    torch.cuda.empty_cache() 

    wait = 0

    best_epoch = 0

    best = np.Inf  

    patience = 10

    for epoch in range(num_epochs):

        # setup progress bar

        sys.stdout.write("[")

        sys.stdout.flush()  

        mynet.train()

        for tn in trainlabeleddata_loader:

            mynet.zero_grad()

            inputs, labels = tn[0].to(device), tn[1].to(device)

            output = mynet(inputs)

            netloss = loss_function(torch.squeeze(output), labels)

            netloss.backward()

            optimizer.step()

#             scheduler.step()

            sys.stdout.write("-")

            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar

        mynet.eval()

        predtrain = []

        for traindata in train_loader:

            traindata = traindata.to(device)

            predtrain.append(torch.squeeze(mynet(traindata)).detach().cpu().numpy())

        predtrain = np.concatenate(predtrain) 

        predvalid = []

        for validdata in valid_loader:

            validdata = validdata.to(device)

            predvalid.append(torch.squeeze(mynet(validdata)).detach().cpu().numpy())

        predvalid = np.concatenate(predvalid) 

        losses.append(log_loss_metric(train_y, expit(predtrain).astype(np.float64)))

        vallosses.append(log_loss_metric(valid_y, expit(predvalid).astype(np.float64)))

        print(f"Epoch: {epoch}, Loss: {losses[epoch]}, LossVal: {vallosses[epoch]}")

        # Early stoping module

        if np.less(vallosses[epoch], best):

            best = vallosses[epoch]

            wait = 0

            best_epoch = epoch

            # Record the best weights if current results is better (less).

            torch.save(mynet.state_dict(), 'checkpoint.pt')

        else:

            wait += 1

            if wait >= patience:

                print("=======================================")

                print("Restoring model weights from the end of the best epoch.")

                mynet.load_state_dict(torch.load('checkpoint.pt'))

                print("Epoch %05d: early stopping" % (best_epoch))

                break

    mynet.eval()

    predvalid = []

    for validdata in valid_loader:

        validdata = validdata.to(device)

        predvalid.append(torch.squeeze(mynet(validdata)).detach().cpu().numpy())

    predvalid = np.concatenate(predvalid) 

    oof[v_idx,:] = predvalid

    predtest = []

    for testdata in test_loader:

        testdata = testdata.to(device)

        predtest.append(torch.squeeze(mynet(testdata)).detach().cpu().numpy())

    predtest = np.concatenate(predtest) 

    predtest = expit(predtest)

    predictions += predtest/nfolds
print('valodation score: {}'.format(log_loss_metric(targets, expit(oof).astype(np.float64))))
# plt.plot(losses)

# plt.plot(vallosses)

# plt.legend(['loss_train','loss_valid'])

# plt.show()
sub = pd.read_csv('../input/lish-moa/sample_submission.csv')

sub.iloc[:,1:] = predictions

sub.loc[test_type=='ctl_vehicle',1:] = 0

sub.to_csv('submission.csv', index=False)