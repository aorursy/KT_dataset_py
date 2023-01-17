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
import matplotlib.pyplot as plt

%matplotlib inline
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
print(train_features.shape) # (23814, 876)

print(train_targets.shape) # (23814, 207)

print(test_features.shape) # (3982, 876)
def preprocess(df):

    """Returns preprocessed data frame"""

    df = df.copy()

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})

    del df['sig_id']

    return df
train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']
train_targets = train_targets.loc[train['cp_type']==0].reset_index(drop=True)

train = train.loc[train['cp_type']==0].reset_index(drop=True)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(train)



# scale train data

train = scaler.transform(train)

# scale test data

test = scaler.transform(test)
targets = [col for col in train_targets.columns]
print(train.shape) # (21948, 875)

print(test.shape) # (3982, 875)

print(train_targets.shape) # (21948, 206)
class MoADataset:

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return self.features.shape[0]

    

    def __getitem__(self, idx):

        return {

            'input': torch.tensor(self.features[idx, :], dtype=torch.float),

            'target': torch.tensor(self.targets[idx, :], dtype=torch.float)

        }

    

class TestDataset:

    def __init__(self, features):

        self.features = features

    

    def __len__(self):

        return self.features.shape[0]

    

    def __getitem__(self, idx):

        return {

            'input': torch.tensor(self.features[idx, :], dtype=torch.float)

        }
import torch

import torch.nn as nn

import torch.nn.functional as F
class Model(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features)

        self.dropout1 = nn.Dropout(0.2) # 0.2

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(0.2) # 0.2

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.dropout3 = nn.Dropout(0.2) # 0.2

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.weight = torch.tensor([0.5]).to(device)

        

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        

        x = F.prelu(self.dense1(x), self.weight) # relu -> prelu

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.prelu(self.dense2(x), self.weight) # relu -> prelu

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = self.dense3(x)

        

        return x
#train = train.values

#test = test.values

train_targets = train_targets.values
def set_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
def get_dataloaders(num_workers, batch_size, x_train, y_train, x_valid, y_valid):

    """Return training and valid dataloader"""

    

    # load the training and valid datasets

    train_dataset = MoADataset(x_train, y_train)

    valid_dataset = MoADataset(x_valid, y_valid)



    # prepare data loaders

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)



    # define loaders

    loader = {

        "train": train_loader,

        "valid": valid_loader

    }

    

    return loader
def get_testloaders(num_workers, batch_size, x_test):

    """Return test dataloader"""

    

    # load the test datasets

    test_dataset = TestDataset(x_test)

    

    # prepare test loader

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    

    # define loaders

    loader = {

        'test': test_loader

    }

    return loader
from torch.optim.lr_scheduler import StepLR



def train_model(n_epochs, loaders, model, optimizer, criterion, device, save_path):

    """Returns a trained model"""        

    scheduler = StepLR(optimizer, step_size=2, gamma=0.96)

    

    # initialize tracker for minimum validation loss

    valid_loss_min = np.Inf

    print(valid_loss_min)

    for epoch in range(1, n_epochs + 1):

        # decay Learning Rate

        scheduler.step()

        # print(f'Epoch: \t{epoch}\tLR: {scheduler.get_lr()}')

        

        # initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        

        # train the model

        model.train()

        

        #for batch_idx, (data, target) in enumerate(loaders['train']):

        for data in loaders['train']:

            data_input, data_target = data['input'].to(device), data['target'].to(device)

            

            # initialize weights to zero: clear the gradients of all optimized variables

            optimizer.zero_grad()

            

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data_input)



            # calcuate loss

            loss = criterion(output, data_target)

            

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            

            # perform a single optimization step

            optimizer.step()

            

            # TODO: scheduler.step()

            

            # update running training loss

            # print("train loss : ", loss.item())

            train_loss += (loss.item() / len(loaders['train']))



        # validate the model

        model.eval()

        

        for data in loaders['valid']:

            data_input, data_target = data['input'].to(device), data['target'].to(device)

            

            # update the average validation loss

            output = model(data_input)

            

            # calculate loss

            loss = criterion(output, data_target)

            

            # update running validation loss

            # print("validation loss : ", loss.item())

            valid_loss += (loss.item() / len(loaders['valid']))

        

        # print training/validation statistics

        # print(f'Epoch: \t{epoch}\tTraining Loss: {train_loss}\tValidation Loss:{valid_loss}')

        

        # save the model if validation loss has descrased

        if valid_loss < valid_loss_min:

            print(f'Epoch: \t{epoch}\tValidation loss decreased ({valid_loss_min} -> {valid_loss}). Saving the model...')

            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss



    # return trained model

    return model
def run_training(seed, kfold, batch_size, epochs, learning_rate, weight_decay):

    set_seed(seed)



    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=train, y=train_targets)):

        x_train, x_valid = train[train_idx], train[valid_idx]

        y_train, y_valid = train_targets[train_idx], train_targets[valid_idx]

        

        # get dataloaders

        dataloaders = get_dataloaders(0, batch_size, x_train, y_train, x_valid, y_valid)

        

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Model(875, 206, 1024).to(device)

        

        criterion_moa = nn.BCEWithLogitsLoss() # for multi-lable classfication

        optimizer_moa = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # train the model

        train_model(epochs, dataloaders, model, optimizer_moa, criterion_moa, device, f'models/model_seed_{seed}_fold_{fold}.pt')
# hyper parameters



FOLDS = 5

WORKERS = 0

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 0.0002

WEIGHT_DECAY = 0.00001

SEED = 42
%mkdir models
mskf = MultilabelStratifiedKFold(n_splits=FOLDS)



for seed in range(40, 45):

    run_training(seed, mskf, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
def inference(loaders, model, device):

    """Return a prediction"""

    

    model.eval()

    preds = []

    

    for data in  loaders['test']:

        data_input = data['input'].to(device)

        

        # forward pass: compute predicted outputs by passing inputs to the model

        with torch.no_grad():

            output = model(data_input)

        

        pred = output.sigmoid().detach().cpu().numpy()

        preds.append(pred)

        

    return np.concatenate(preds)
def run_inferencing(seed):

    set_seed(seed)

    

    preds = np.zeros((len(test), 206))

    

    #for fold, (train_idx, valid_idx) in enumerate(mskf.split(X=train, y=train_targets)):

    for i in range(0, FOLDS):    

        # get dataloaders

        dataloaders = get_testloaders(0, 128, test)

        

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Model(875, 206, 1024)

        

        model.load_state_dict(torch.load(f'models/model_seed_{seed}_fold_{i}.pt'))

        model.to(device)

        

        pred = inference(dataloaders, model, device)

        

        preds += pred

        

    preds = preds / FOLDS

    return preds
#preds = run_inferencing(SEED)
preds = np.zeros((len(test), 206))

for seed in range(40, 45):

    preds += run_inferencing(seed)

preds = preds / 5
sample_submission[targets] = preds

sample_submission.loc[test_features['cp_type']=='ctl_vehicle', targets] = 0

sample_submission.to_csv('submission.csv', index=False)
sample_submission