import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import os

import copy

import seaborn as sns



from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import warnings

warnings.filterwarnings('ignore')
# # My

# import numpy as np

# import random

# import pandas as pd

# import matplotlib.pyplot as plt

# import os

# import copy

# import seaborn as sns

# from sklearn import preprocessing

# from sklearn.metrics import log_loss

# from sklearn.preprocessing import StandardScaler

# from sklearn.decomposition import PCA





# import torch

# import torch.nn as nn

# import torch.nn.functional as F

# import torch.optim as optim



# import warnings 

# warnings.filterwarnings('ignore')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
#My 

print(train_features.shape)

print(test_features.shape)
# #My

# GENES = [col for col in train_features.columns if col.startswith('g-')]

# CELLS = [col for col in train_features.columns if col.startswith('c-')]
GENES = [col for col in train_features.columns if col.startswith('g-')]

CELLS = [col for col in train_features.columns if col.startswith('c-')]
def seed_everything(seed=1903):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(seed=1903)
# GENES

n_comp = 28



data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

data2 = (PCA(n_components=n_comp, random_state=1903).fit_transform(data[GENES]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)



#CELLS

n_comp = 5



data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

data2 = (PCA(n_components=n_comp, random_state=1903).fit_transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)
print(train_features.shape)

print(test_features.shape)
#My

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
#My

train_features.head()
# #My 

# n_comp = 28

# data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

# pca = PCA(n_components=n_comp, random_state=1903)

# data2 = pca.fit_transform(data[GENES])

# train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



# train2 = pd.DataFrame(train2, columns = [f'pca_G-{i}' for i in range(n_comp)])

# test2 = pd.DataFrame(test2, columns = [f'pca_G-{i}' for i in range(n_comp)])



# train_features = pd.concat([train_features, train2], axis = 1)

# test_features = pd.concat([test_features, test2], axis =1)

# print(train_features.shape)

# print(test_features.shape)
# #My

# n_comp = 5

# data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

# pca = PCA(n_components=n_comp, random_state=1903)

# data2 = pca.fit_transform(data[CELLS])

# train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



# train2 = pd.DataFrame(train2, columns = [f'pca_C-{i}' for i in range(n_comp)])

# test2 = pd.DataFrame(test2, columns = [f'pca_C-{i}' for i in range(n_comp)])



# train_features = pd.concat([train_features, train2], axis = 1)

# test_features = pd.concat([test_features, test2], axis =1)

# print(train_features.shape)

# print(test_features.shape)
# #My

# from sklearn.feature_selection import VarianceThreshold

# var_thresh = VarianceThreshold(threshold=0.4)

# data = train_features.append(test_features)

# data_transformed = data.iloc[:, 4:]

# data_transformed.shape

# var_thresh.fit_transform(data_transformed)

# data_transformed = data_transformed[data_transformed.columns[var_thresh.get_support(indices=True)]]

# data_transformed.head()

# #My

# from sklearn.feature_selection import VarianceThreshold

# var_thresh = VarianceThreshold(threshold=0.4)

# data = train_features.append(test_features)

# orig_cols = data.columns

# print(data.shape)

# var_thresh.fit_transform(data.iloc[:, 4:])

# # data = data[data.columns[var_thresh.get_support(indices=True)]]

# data = pd.concat([data.iloc[:, :4], data[data.iloc[:, 4:].columns[var_thresh.get_support(indices=True)]] ], axis = 1)

# reduced_cols = data.columns

# print(data.shape)

# print("Removed columns", set(orig_cols)-set(reduced_cols))

# # data.head()

# # sel_var=sel.fit_transform(data)

# # data[data.columns[sel.get_support(indices=True)]] 
data.head()
from sklearn.feature_selection import VarianceThreshold





var_thresh = VarianceThreshold(threshold=0.4)

data = train_features.append(test_features)

data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])



train_features_transformed = data_transformed[ : train_features.shape[0]]

test_features_transformed = data_transformed[-test_features.shape[0] : ]





train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                              columns=['sig_id','cp_type','cp_time','cp_dose'])



train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)





test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                             columns=['sig_id','cp_type','cp_time','cp_dose'])



test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)



train_features
train_features.shape
# #My

# train = train_features.merge(train_targets_scored, on='sig_id')

# train = train[train['cp_type'] != 'ctl_vechile'].reset_index(drop=True)



# test = test_features[test_features['cp_type'] != 'ctl_vechile'].reset_index(drop=True)



# target = train[train_targets_scored.columns]



# train = train.drop('cp_type', axis=1)

# test = test.drop('cp_type', axis=1)
train = train_features.merge(train_targets_scored, on='sig_id')

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)



target = train[train_targets_scored.columns]



train = train.drop('cp_type', axis=1)

test = test.drop('cp_type', axis=1)

# #My

# target_cols = list(target.drop('sig_id', axis=1).columns.values)

# #My

# target_cols = target.drop('sig_id', axis=1).columns.values.tolist
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
# #My

# folds = train.copy()

# mskf = MultilabelStratifiedKFold(n_splits=7)



# for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):

#     folds.loc[v_idx, 'kfold'] = int(f)

# folds['kfold'] = folds['kfold'].astype(int)
folds = train.copy()



mskf = MultilabelStratifiedKFold(n_splits=7)



for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):

    folds.loc[v_idx, 'kfold'] = int(f)



folds['kfold'] = folds['kfold'].astype(int)
# #My



# class MoaDataset:

#     def __init__(self, features, targets):

#         self.features = features

#         self.targets = targets

    

#     def __len__(self):

#         return self.features.shape[0]

    

#     def __getitem__(self, idx):

#         dct = {

#             'x': torch.tensor(self.features[idx,:], dtype=torch.float),

#             'y': torch.tensor(self.features[idx,:], dtype=torch.float)

#         }

#         return dct

    

    

# class TestDataset:

#     def __init__(self, features):

#         self.features = features

    

#     def __len__(self):

#         return self.features.shape[0]

    

#     def __getitem__(self, idx):

#         dct = {

#             'x': torch.tensor(self.features[idx, :], dtype=torch.float)

#         }
class MoADataset:

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),

            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            

        }

        return dct

    

class TestDataset:

    def __init__(self, features):

        self.features = features

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)

        }

        return dct
# #My

# def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):

#     model.train()

#     final_loss = 0

    

#     for data in dataloader:

#         optimizer.zero_grad()

#         inputs, targets = data['x'].to(device), data['y'].to(device)

#         outputs = model(inputs)

#         loss = loss_fn(outputs, targets)

#         loss.backward()

#         optimizer.step()

#         scheduler.step()

        

#         final_loss  += loss.item()

#     final_loss /= len(dataloader)

#     return final_loss



# def valid_fn(model, loss_fn, dataloader, device):

#     model.eval()

#     final_loss = 0

#     valid_preds = []

    

#     for data in dataloader:

#         inputs, targets = data['x'].to(device), data['y'].to(device)

#         outputs = model(inputs)

#         loss = loss_fn(outputs, targets)

        

                

#         final_loss += loss.item()

#         valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        

#     final_loss /= len(dataloader)

#     valid_preds = np.concatenate(valid_preds)

    

#     return final_loss, valid_preds



# def inference_fn(model, dataloader, device):

#     model.eval()

#     preds = []

    

#     for data in dataloader:

#         inputs = data['x'].to(device)



#         with torch.no_grad():

#             outputs = model(inputs)

        

#         preds.append(outputs.sigmoid().detach().cpu().numpy())

        

#     preds = np.concatenate(preds)

    

#     return preds
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):

    model.train()

    final_loss = 0

    

    for data in dataloader:

        optimizer.zero_grad()

        inputs, targets = data['x'].to(device), data['y'].to(device)

        #print(inputs.shape)

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        loss.backward()

        optimizer.step()

        scheduler.step()

        

        final_loss += loss.item()

        

    final_loss /= len(dataloader)

    

    return final_loss





def valid_fn(model, loss_fn, dataloader, device):

    model.eval()

    final_loss = 0

    valid_preds = []

    

    for data in dataloader:

        inputs, targets = data['x'].to(device), data['y'].to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        

        final_loss += loss.item()

        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    final_loss /= len(dataloader)

    valid_preds = np.concatenate(valid_preds)

    

    return final_loss, valid_preds



def inference_fn(model, dataloader, device):

    model.eval()

    preds = []

    

    for data in dataloader:

        inputs = data['x'].to(device)



        with torch.no_grad():

            outputs = model(inputs)

        

        preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    preds = np.concatenate(preds)

    

    return preds

   
# #My

# class Model(nn.Module):

#     def __init__(self, num_features, num_targets, hidden_size):

#         super().__init__()

#         self.batch_norm1 = nn.BatchNorm1d(num_features)

#         self.dropout1 = nn.Dropout(0.2)

#         self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        

#         self.batch_norm2 = nn.BatchNorm1d(hidden_size)

#         self.dropout2 = nn.Dropout(0.3)

#         self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        

#         self.batch_norm3 = nn.BatchNorm1d(hidden_size)

#         self.dropout3 = nn.Dropout(0.25)

#         self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

        

#     def forward(self, x):

#         x = self.batch_norm1(x)

#         x = self.dropout1(x)

#         x = F.relu(self.dense1(x))

        

#         x = self.batch_norm2(x)

#         x = self.dropout2(x)

#         x = F.relu(self.dense2(x))

        

#         x = self.batch_norm3(x)

#         x = self.dropout3(x)

#         x = self.dense3(x)
class Model(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(0.3)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.dropout3 = nn.Dropout(0.25)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = self.dense3(x)

        

        return x
# #My

# def process_data(data):

#     data = pd.get_dummies(data, columns = ['cp_time', 'cp_dose'])

#     return data
def process_data(data):

    

    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])

#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})

#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})



# --------------------- Normalize ---------------------

#     for col in GENES:

#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))

    

#     for col in CELLS:

#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))

    

#--------------------- Removing Skewness ---------------------

#     for col in GENES + CELLS:

#         if(abs(data[col].skew()) > 0.75):

            

#             if(data[col].skew() < 0): # neg-skewness

#                 data[col] = data[col].max() - data[col] + 1

#                 data[col] = np.sqrt(data[col])

            

#             else:

#                 data[col] = np.sqrt(data[col])

    

    return data
# #My 

# feature_cols = [col for col in process_data(folds).columns if col not in target_cols]

# feature_cols = [col for col in feature_cols not in ['kfold', 'sig_id']]

# print(len(feature_cols))
feature_cols = [c for c in process_data(folds).columns if c not in target_cols]

feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]

len(feature_cols)
# #My



# #HyperParameters



# DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# EPOCHS = 25

# BATCH_SIZE = 128

# LEARNING_RATE = 1e-3

# WEIGHT_DECAY = 1e-5

# NFOLDS = 7

# EARLY_STOPPING_STEPS = 10

# EARLY_STOP = True



# num_features = len(feature_cols)

# num_targets = len(target_cols)

# hidden_size = 2048
# HyperParameters



DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 25

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

NFOLDS = 7

EARLY_STOPPING_STEPS = 10

EARLY_STOP = True



num_features=len(feature_cols)

num_targets=len(target_cols)

hidden_size=2048
# #My

# def run_training(fold, seed):

#     seed_everything(seed)

    

#     train = process_data(folds)

#     test_ = process_data(test)

    

#     trn_idx = train[train['kfold'] != fold].index

#     val_idx = train[train['kfold'] == fold].index

    

#     train_df = train[train['kfold'] != fold].reset_index(drop=True)

#     valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    

#     x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values

#     x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

    

#     train_dataset = MoADataset(x_train, y_train)

#     valid_dataset = MoADataset(x_valid, y_valid)

#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#     validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    

#     model = Model(

#         num_features=num_features,

#         num_targets=num_targets,

#         hidden_size=hidden_size,

#     )

    

    

#     model.to(DEVICE)

#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,

#                                               max_lr=1e-2, epochs=EPOCHS, step_per_epoch=len(trainloader))

#     loss_fn = nn.BECWithLogitsLoss()

    

#     early_stopping_steps = EARLY_STOPPING_STEPS

#     early_step = 0

    

#     oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))

#     best_loss = np.inf

    

#     for epoch in range(EPOCHS):

#         train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)

#         print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

#         valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

#         print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

        

#         if valid_loss < best_loss:

#             best_loss = valid_loss

#             oof[val_idx] = valid_preds

#             torch.save(model.state_dict(), f"FOLD{fold}_.pth")

#         elif(EARLY_STOP == True):

            

#             early_step += 1

#             if (early_step >= early_stopping_steps):

#                 break

    

    

    

    



    
def run_training(fold, seed):

    

    seed_everything(seed)

    

    train = process_data(folds)

    test_ = process_data(test)

    

    trn_idx = train[train['kfold'] != fold].index

    val_idx = train[train['kfold'] == fold].index

    

    train_df = train[train['kfold'] != fold].reset_index(drop=True)

    valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    

    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values

    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values

    

    train_dataset = MoADataset(x_train, y_train)

    valid_dataset = MoADataset(x_valid, y_valid)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    

    model = Model(

        num_features=num_features,

        num_targets=num_targets,

        hidden_size=hidden_size,

    )

    

    model.to(DEVICE)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 

                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    

    loss_fn = nn.BCEWithLogitsLoss()

    

    early_stopping_steps = EARLY_STOPPING_STEPS

    early_step = 0

    

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))

    best_loss = np.inf

    

    for epoch in range(EPOCHS):

        

        train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)

        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

        

        if valid_loss < best_loss:

            

            best_loss = valid_loss

            oof[val_idx] = valid_preds

            torch.save(model.state_dict(), f"FOLD{fold}_.pth")

        

        elif(EARLY_STOP == True):

            

            early_step += 1

            if (early_step >= early_stopping_steps):

                break

            

    

    #--------------------- PREDICTION---------------------

    x_test = test_[feature_cols].values

    testdataset = TestDataset(x_test)

    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    

    model = Model(

        num_features=num_features,

        num_targets=num_targets,

        hidden_size=hidden_size,

    )

    

    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))

    model.to(DEVICE)

    

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))

    predictions = inference_fn(model, testloader, DEVICE)

    

    return oof, predictions
def run_k_fold(NFOLDS, seed):

    oof = np.zeros((len(train), len(target_cols)))

    predictions = np.zeros((len(test), len(target_cols)))

    

    for fold in range(NFOLDS):

        oof_, pred_ = run_training(fold, seed)

        

        predictions += pred_ / NFOLDS

        oof += oof_

        

    return oof, predictions
# Averaging on multiple SEEDS



SEED = [1903, 1881]

oof = np.zeros((len(train), len(target_cols)))

predictions = np.zeros((len(test), len(target_cols)))



for seed in SEED:

    

    oof_, predictions_ = run_k_fold(NFOLDS, seed)

    oof += oof_ / len(SEED)

    predictions += predictions_ / len(SEED)



train[target_cols] = oof

test[target_cols] = predictions
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)





y_true = train_targets_scored[target_cols].values

y_pred = valid_results[target_cols].values



score = 0

for i in range(len(target_cols)):

    score_ = log_loss(y_true[:, i], y_pred[:, i])

    score += score_ / target.shape[1]

    

print("CV log_loss: ", score)
sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

sub.to_csv('submission.csv', index=False)