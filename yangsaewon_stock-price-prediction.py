%reload_ext autoreload

%autoreload 2

%reload_ext line_profiler

%matplotlib inline



import gc

import os

import time

import math

import random

import numexpr

import itertools

import numpy as np

import pandas as pd

from pathlib import Path

from collections import Counter



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix



import torch

from torch import nn, cuda

from torch.nn import functional as F

import torchvision.models as models

from torch.utils.data import DataLoader, Dataset



from torch.optim import Adam, SGD, Optimizer

from torch.optim.lr_scheduler import ReduceLROnPlateau



plt.style.use('fivethirtyeight')

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import warnings

warnings.filterwarnings('ignore')
# 실험의 재생산을 위한 seed값 고정 함수

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
# 라벨 인코딩 (train과 test의 종목 합쳐서 진행)

def encode_LE(col, train, test, verbose=True):

    df_comb = pd.concat([train[col],train[col]],axis=0)

    df_comb,_ = df_comb.factorize(sort=True)

    nm = col + '_encoded'

    if df_comb.max() > 32000: 

        train[nm] = df_comb[:len(train)].astype('int32')

        test[nm] = df_comb[len(train):].astype('int32')

    else:

        train[nm] = df_comb[:len(train)].astype('int16')

        test[nm] = df_comb[len(train):].astype('int16')

    del df_comb; x = gc.collect()

    if verbose: print(nm)
# Merge Column을 기준으로 agg하는 함수 - ex) code (종목)별로 평균 F 피쳐 값을 계산으로써 파생 변수를 생성해 모델에 종목 정보를 제공해줄 수 있다

def code_agg(train_df, test_df, merge_columns, columns, aggs=['mean']):

    tr, te = df_copy(train_df, test_df)

    for merge_column in merge_columns:

        for col in columns:

            for agg in aggs:

                valid = pd.concat([tr[[merge_column, col]], te[[merge_column, col]]])

                new_cn = merge_column + '_' + agg + '_' + col

                if agg=='quantile':

                    valid = valid.groupby(merge_column)[col].quantile(0.8).reset_index().rename(columns={col:new_cn})

                else:

                    valid = valid.groupby(merge_column)[col].agg([agg]).reset_index().rename(columns={agg:new_cn})

                valid.index = valid[merge_column].tolist()

                valid = valid[new_cn].to_dict()

            

                tr[new_cn] = tr[merge_column].map(valid)

                te[new_cn] = te[merge_column].map(valid)

    return tr, te
def df_copy(tr_df, te_df):

    tr = tr_df.copy()

    te = te_df.copy()

    return tr, te
# use numexpr library for improved performance

# Simple Moving Average (code 별로 계산)

def SMA(df, target, num_windows=3):

    arr = np.array([])

    x = df['code_encoded'].values

    for code in df['code_encoded'].unique():

        temp_df = df[numexpr.evaluate(f'(x == code)')]

        arr = np.concatenate((arr, temp_df[target].rolling(window=num_windows, min_periods=1).mean().values))

    return arr



# Exponential Moving Average (code 별로 계산)

def EMA(df, target, span_num=3):

    arr = np.array([])

    x = df['code_encoded'].values

    for code in df['code_encoded'].unique():

        temp_df = df[numexpr.evaluate(f'(x == code)')]

        arr = np.concatenate((arr, temp_df[target].ewm(span=span_num, min_periods=1).mean().values))

    return arr
# 데이터셋 NaN 전처리

def preprocess_nan(df, feat_cols):



    preprocessed_df = pd.DataFrame()



    # code(종목)별, 피쳐별로 NaN 값을 채우되, 최근 시간에 더 가중치를 두고 계산한다.

    for code in df['code'].unique():

        code_df = df[df['code'] == code].copy()



        for i, feat_col in enumerate(code_df[feat_cols]):



            temp_df = code_df[feat_col].dropna().to_frame()



            if len(temp_df) == len(code_df):

                continue



            temp_df['weight'] = [i for i in range(len(temp_df), 0, -1)]



            try:

                fill_value = np.average(temp_df[feat_col], weights=temp_df['weight'])

                code_df[feat_col].fillna(fill_value, inplace=True)



            except:

                continue



        preprocessed_df = preprocessed_df.append(code_df)

        

    del code_df, temp_df; gc.collect()

    

    # 해당 code(종목)의 feature에 모든 값이 NaN일 경우 모든 종목의 피쳐 평균값으로 NaN을 채운다.  

    for feat_col in feat_cols:

        mean_val = preprocessed_df[feat_col].mean()

        preprocessed_df[feat_col].fillna(mean_val, inplace=True)



    return preprocessed_df
# 예측값이 0.5 이상일 경우 1, 0.5 이하일 경우 0으로 mapping한다.

# threshold는 수정 가능

def to_binary(preds, threshold=0.5):

    return np.where(preds >= threshold, 1, 0)
# 예측 결과를 기반으로 Confusion Matrix 시각화하는 함수

def plot_confusion_matrix(cm, target_names, title='Validation Confusion matrix', cmap=None, normalize=True):



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('OrRd_r')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="black" if cm[i, j] > thresh else "white")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="black" if cm[i, j] > thresh else "white")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
# 단위 시간 td를 기준으로 validation set을 생성하는 위한 함수

def make_dict(train_df, test_df, feat_cols, target):



    dataset_dict = {}



    # dataset 1

    X_train1 = train_df.query("td <= 172")[feat_cols].values

    X_valid1 = train_df.query("td > 172 & td <= 206")[feat_cols].values

    

    scaler = StandardScaler()

    scaler.fit(X_train1)

    

    dataset_dict['X_train1'] = scaler.transform(X_train1)

    dataset_dict['X_valid1'] = scaler.transform(X_valid1)

    dataset_dict['y_train1'] = train_df.query("td <= 172")[target].values

    dataset_dict['y_valid1'] = train_df.query("td > 172 & td <= 206")[target].values

    del scaler



    # dataset 2

    X_train2 = train_df.query("td <= 206")[feat_cols].values

    X_valid2 = train_df.query("td > 206 & td <= 240")[feat_cols].values

    

    scaler = StandardScaler()

    scaler.fit(X_train2)

    

    dataset_dict['X_train2'] = scaler.transform(X_train2)

    dataset_dict['X_valid2'] = scaler.transform(X_valid2)

    dataset_dict['y_train2'] = train_df.query("td <= 206")[target].values

    dataset_dict['y_valid2'] = train_df.query("td > 206 & td <= 240")[target].values

    del scaler



    # dataset 3

    X_train3 = train_df.query("td <= 240")[feat_cols].values

    X_valid3 = train_df.query("td > 240 & td <= 274")[feat_cols].values

    

    scaler = StandardScaler()

    scaler.fit(X_train3)

    

    dataset_dict['X_train3'] = scaler.transform(X_train3)

    dataset_dict['X_valid3'] = scaler.transform(X_valid3)

    dataset_dict['y_train3'] = train_df.query("td <= 240")[target].values

    dataset_dict['y_valid3'] = train_df.query("td > 240 & td <= 274")[target].values



    x_test = test_df[feat_cols].values

    dataset_dict['x_test'] = scaler.transform(x_test)



    return dataset_dict
# 데이터를 받아 학습을 하는 함수. validation 방법에 따라 다르며, feature importance와 confusion matrix도 시각화해 볼 수 있다.

def make_predictions(dataset_dict, df, feat_cols, lgb_params, valid_type='hold_out', plot_importance=True, plot_confusion=True):



    x_test = dataset_dict['x_test']

    if valid_type == 'hold_out':

        X_train = dataset_dict['X_train3']

        y_train = dataset_dict['y_train3']

        X_valid = dataset_dict['X_valid3']

        y_valid = dataset_dict['y_valid3']

        

        trn_data = lgb.Dataset(X_train, label=y_train)

        val_data = lgb.Dataset(X_valid, label=y_valid) 



        clf = lgb.train(

            lgb_params,

            trn_data,

            valid_sets = [trn_data, val_data],

            verbose_eval = 100 ,

        )   

        

        valid_preds = clf.predict(X_valid)

        preds = clf.predict(x_test)

        

        print()

        print("roc_auc_score: {:.4f}".format(roc_auc_score(y_valid, valid_preds)))

        print("accuracy_score: {:.4f}".format(accuracy_score(y_valid, to_binary(valid_preds, 0.5))))



        if plot_importance:

            

            feature_importance_df = pd.DataFrame()

            feature_importance_df["Feature"] = feat_cols

            feature_importance_df["Importance"] = clf.feature_importance()

            

            cols = (feature_importance_df[["Feature", "Importance"]]

                .groupby("Feature")

                .mean()

                .sort_values(by="Importance", ascending=False).index)



            best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



            plt.figure(figsize=(14,10))

            sns.barplot(x="Importance",

                        y="Feature",

                        data=best_features.sort_values(by="Importance",

                                                       ascending=False)[:20], ci=None)

            plt.title('LightGBM Feature Importance', fontsize=20)

            plt.tight_layout()

            

        if plot_confusion:

            plot_confusion_matrix(confusion_matrix(y_valid, to_binary(valid_preds, 0.5)), 

                          normalize    = False,

                          target_names = ['pos', 'neg'],

                          title        = "Confusion Matrix")

    

    elif valid_type == 'sliding_window':

        

        window_num = 3

        acc = 0

        auc = 0

        for num in range(1, window_num+1):    

            print(f"num {num} dataset training starts")

            

            preds = np.zeros(len(x_test))

            X_train = dataset_dict[f'X_train{num}']

            y_train = dataset_dict[f'y_train{num}']

            X_valid = dataset_dict[f'X_valid{num}']

            y_valid = dataset_dict[f'y_valid{num}']

            trn_data = lgb.Dataset(X_train, label=y_train)

            val_data = lgb.Dataset(X_valid, label=y_valid)



            clf = lgb.train(lgb_params, trn_data, valid_sets = [trn_data, val_data], verbose_eval=100)

            valid_preds = clf.predict(X_valid)

            preds += clf.predict(x_test) / window_num

            acc += accuracy_score(y_valid, to_binary(valid_preds, 0.5)) / window_num

            auc += roc_auc_score(y_valid, valid_preds) / window_num

            print()            

        

        print("mean roc_auc_score: {:.4f}".format(auc))

        print("mean acc_score: {:.4f}".format(acc))



    return preds
# improved version of adam optimizer

class AdamW(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay)

        super(AdamW, self).__init__(params, defaults)



    def step(self, closure=None):

        """Performs a single optimization step.



        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')



                state = self.state[p]



                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                # according to the paper, this penalty should come after the bias correction

                # if group['weight_decay'] != 0:

                #     grad = grad.add(group['weight_decay'], p.data)



                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)



                denom = exp_avg_sq.sqrt().add_(group['eps'])



                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1



                p.data.addcdiv_(-step_size, exp_avg, denom)



                if group['weight_decay'] != 0:

                    p.data.add_(-group['weight_decay'], p.data)



        return loss
# Multi-Layer Perceptron baseline model (further improvements needed)

class MLP(nn.Module):

    def __init__(self, num_classes=1, num_feats=47):

        super(MLP, self).__init__()

        self.num_classes = num_classes         

        self.mlp_layers = nn.Sequential(

            nn.Linear(num_feats, 1024),

            nn.PReLU(),

            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),

            nn.PReLU(),

            nn.BatchNorm1d(512),

            nn.Linear(512, 256),

            nn.PReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.PReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.2),

            nn.Linear(128, self.num_classes)

        )

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()



    def forward(self, x):

        out = self.mlp_layers(x)

        return self.sigmoid(out)



    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)
# numpy array를 받아 dataset으로

class Stock_mlp_dataset(Dataset):

    def __init__(self, X, Y):

        self.X = X

        self.Y = Y

        self.X_dataset = []

        self.Y_dataset = []

        for x in X:

            self.X_dataset.append(torch.FloatTensor(x))

        try:

            for y in Y:

                self.Y_dataset.append(torch.tensor(y))

        

        # test set의 경우엔 라벨이 없음

        except:

#             print("no_label")

            pass



    def __len__(self):

        return len(self.X)



    def __getitem__(self, index):

        inputs = self.X_dataset[index]

        # train, valid set

        try:

            target = self.Y_dataset[index]

            return inputs, target

        # test set

        except:

            return inputs
# torch dataset load 함수 (dataset => dataloader 반환)

def build_dataloader(X, Y, batch_size, shuffle=False):

    

    dataset = Stock_mlp_dataset(X, Y)

    dataloader = DataLoader(

                            dataset,

                            batch_size=batch_size,

                            shuffle=shuffle,

                            num_workers=2

                            )

    return dataloader
# model build 함수 

def build_model(device, model_name='MLP', num_classes=1, num_feats=47):

    if model_name == 'MLP':

        model = MLP(num_classes, num_feats)

#     모델 추가 가능

#     elif model_name == '':

#         model = _

    else:

        raise NotImplementedError

    model.to(device)

    return model
# 매 epoch마다 validation을 진행, 각종 metric의 score를 반환

def validation(model, criterion, valid_loader, use_cuda):

    

    model.eval()

    

    valid_preds = []

    valid_targets = []

    val_loss = 0.

    

    with torch.no_grad():

        for batch_idx, (inputs, target) in enumerate(train_loader):



            target = target.reshape(-1, 1).float()

            valid_targets.append(target.numpy().copy())



            if use_cuda:

                inputs = inputs.to(device)

                target = target.to(device)       

                    

            output = model(inputs)

            

#             print(output[:10])

#             print(target[:10])



            loss = criterion(output, target)

            valid_preds.append(output.detach().cpu().numpy())

            

            val_loss += loss.item() / len(valid_loader)

     

    # to_binary 함수를 통해 0과 1 사이의 아웃풋을 0과 1로 매핑

    valid_preds = np.concatenate(valid_preds)

    valid_targets = np.concatenate(valid_targets)

    acc = accuracy_score(valid_targets, to_binary(valid_preds))

    roc_auc = roc_auc_score(valid_targets, valid_preds)

    recall = recall_score(valid_targets, to_binary(valid_preds)) 

    precision = precision_score(valid_targets, to_binary(valid_preds)) 

    

    return val_loss, acc, roc_auc, recall, precision
# MLP 모델을 학습하는 함수

def train_mlp_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, use_cuda, verbose_epoch=20, path='best_model.pt'):

    

    best_valid_acc = 0.0

    best_epoch = 0

    count = 0

    start_time = time.time()

    

    for epoch in range(1, num_epochs+1):



        model.train()

        optimizer.zero_grad()

        train_loss = 0.0



        for batch_idx, (inputs, target) in enumerate(train_loader):



            target = target.reshape(-1, 1).float()



            if use_cuda:

                inputs = inputs.to(device)

                target = target.to(device)



            output = model(inputs)

            loss = criterion(output, target)



            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            train_loss += loss.item() / len(train_loader)



        # validation 진행

        val_loss, acc_score, auc_score, recall, precision = validation(model, criterion, valid_loader, use_cuda)



        elapsed = time.time() - start_time



        lr = [_['lr'] for _ in optimizer.param_groups]



        if epoch % verbose_epoch == 0:

            print('\nEpoch [{}/{}]  train Loss: {:.3f}  val_loss: {:.3f}  accuracy: {:.3f}  roc_auc: {:.3f}  recall: {:.3f}  precision: {:.4f}  lr: {:.7f}  elapsed: {:.0f}m {:.0f}s' \

                  .format(epoch,  num_epochs, train_loss, val_loss, acc_score, auc_score, recall, precision, lr[0], elapsed // 60, elapsed % 60))

        

        model_path = output_dir / path



        # validation accuracy가 최대일 때 모델 저장

        if acc_score > best_valid_acc:

            best_epoch = epoch

            best_valid_acc = acc_score

            roc_auc_score = auc_score

            recall_ = recall

            precision_ = precision

        

            torch.save(model.state_dict(), model_path)



        # 50 epoch 동안 개선이 없으면 학습 강제 종료

        if count == 50:

            print("not improved for 50 epochs")

            break

        if acc_score < best_valid_acc:

            count += 1

        else:

            count = 0

            

        # learning_rate scheduling based on accuracy score

        scheduler.step(acc_score)

            

    print("\n- training report")

    print("best epoch: {}".format(best_epoch))

    print("accuracy: {:.4f}".format(best_valid_acc))

    print("roc_auc_score: {:.4f}".format(roc_auc_score))

    print("recall score: {:.4f}".format(recall_))

    print("precision score: {:.4f}\n".format(precision_))
def mlp_inference(model, test_loader, batch_size, use_cuda):

    

    test_preds = []

    model.eval()

    

    with torch.no_grad():

        for batch_idx, data in enumerate(test_loader):

            if use_cuda:

                data = data.to(device)

            outputs = model(data)

            test_preds.append(outputs.detach().cpu().numpy())

            

    test_preds = np.concatenate(test_preds)

    return test_preds
# lstm 모델 학습 (하나의 code만)

def train_lstm_model(model, data_loader, criterion, num_epochs, verbose_eval, model_path):

    

    optimizer = Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')

    best_epoch = 0

    

    for epoch in range(1, num_epochs+1):



        model.train()

        optimizer.zero_grad()

        train_loss = 0



        for i, (X, Y) in enumerate(data_loader):

            X = X.float()

            Y = Y.float()

            if use_cuda:

                X = X.to(device)

                Y = Y.to(device)

            output = model(X) 



            loss = 0

            preds = []

            for i, y_t in enumerate(Y.chunk(Y.size(1), dim=1)):

                loss += criterion(output[i], y_t)



            loss.backward()

            optimizer.step()

            optimizer.zero_grad()



            train_loss += loss.item() / len(data_loader)



        # train loss가 낮을 때 모델 저장

        if train_loss < best_loss:

            best_epoch = epoch

            best_loss = train_loss

            torch.save(model.state_dict(), model_path)



        if epoch % (verbose_eval) == 0:

            print("Epoch [{}/{}]  train_loss: {:.5f}".format(epoch, num_epochs, train_loss))    



    print("\nBest epoch: {}  Best Loss: {:.5f}".format(best_epoch, best_loss))
# 특정 코드의 데이터를 불러와 scaling => 이후 Stock_lstm_dataset 함수에 인풋으로 전달

def prepared_code_data(df, code, feat_cols, seq_len=12):

    

    temp_df = df[df['code']==code][feat_cols+['target']].reset_index(drop=True)



    X = temp_df[feat_cols].values

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    Y = temp_df['target'].values

    

    return X, Y
# lstm을 위한 데이터셋 생성 함수

class Stock_lstm_dataset(Dataset):

    def __init__(self, X, Y, seq_len):

        self.X = []

        self.Y = []

        for i in range(len(X) - seq_len):

            self.X.append(X[i : i + seq_len])

            self.Y.append(Y[i : i + seq_len])



    def __len__(self):

        return len(self.X)



    def __getitem__(self, index):

        inputs = torch.tensor(self.X[index])

        targets = torch.tensor(self.Y[index])

        return inputs, targets
# simple rnn based model (further improvements needed)

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=False, batch_first=True)

        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False, batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.output_size)



    def forward(self, x):

        outputs = []

        for i, x_t in enumerate(x.chunk(x.size(1), dim=1)):

            h_lstm1, _ = self.lstm(x_t)

            h_lstm2, _ = self.lstm1(h_lstm1)

            output = self.linear(h_lstm2)

            outputs += [output.squeeze(-1)]

        return outputs
# lstm 모델의 inference 함수

def lstm_predict(model, X, seq_len):

    i = 0

    result = []

    while (i < X.shape[0]):



        batch_end = i + seq_len



        if batch_end > X.shape[0]:

            batch_end = X.shape[0]

        x_input = torch.tensor(X[i: batch_end])



        if x_input.dim() == 2:

            x_input = x_input.unsqueeze(0)



        x_input = x_input.float()

        if use_cuda:

            x_input = x_input.to(device)



        output = model(x_input)

        for value in output:

            result.append(value.item())



        i = batch_end

    return result
# 하나의 종목을 학습하고 이를 시각화까지 해주는 함수

def lstm_train_visualize(train_df, code, feat_cols, num_epochs, seq_len, verbose_eval=20, plot_prediction=True):

    X, Y = prepared_code_data(train_df, code, feat_cols, seq_len)



    dataset = Stock_lstm_dataset(X, Y, seq_len=seq_len)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)



    # model setting

    INPUT_SIZE = len(feat_cols)

    HIDDEN_SIZE = 100

    OUTPUT_SIZE = 1



    model = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    model.to(device)



    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=0.0001)



    model_path = output_dir / 'rnn_best_model.pt'



    train_lstm_model(model, data_loader, criterion, num_epochs, verbose_eval, model_path)



    model = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    model.load_state_dict(torch.load(model_path))

    model.to(device)



    model.eval()



    result = lstm_predict(model, X, seq_len)

    result_df = pd.DataFrame({'td':[i for i in range(X.shape[0])], 'predicted':result, 'target':Y})

    

    if plot_prediction:

        

        data = []

        

        data.append(go.Scatter(

            x = result_df['td'].values,

            y = result_df['target'].values,

            name = "target"

        ))

        

        data.append(go.Scatter(

            x = result_df['td'].values,

            y = result_df['predicted'].values,

            name = "predicted"

        ))

        layout = go.Layout(dict(title = f"code: {code}",

                          xaxis = dict(title = 'Time'),

                          yaxis = dict(title = 'Earning Ratio (target)'),

                          ),legend=dict(orientation="h"))



        py.iplot(dict(data=data, layout=layout), filename='basic-line')

# 결과 재생산을 위한 seed값 고정

seed = 42

seed_everything(seed)
DATASET_PATH = '../input/stock-price'



X_train = pd.read_csv(os.path.join(DATASET_PATH, 'train_data.csv')) #훈련 데이터

Y_train = pd.read_csv(os.path.join(DATASET_PATH,'train_target.csv')) # 훈련 데이터에 대한 정답데이터 for regression

Y2_train = pd.read_csv(os.path.join(DATASET_PATH,'train_target2.csv')) # 훈련 데이터에 대한 정답데이터 for classification

test_df = pd.read_csv(os.path.join(DATASET_PATH,'test_data.csv')) # 테스트 데이터
X_train = X_train.set_index(['td', 'code'])

Y_train = Y_train.set_index(['td', 'code'])

Y2_train = Y2_train.set_index(['td', 'code'])
# 시각화 및 전처리부터 모델링까지 보다 편하게 수행하기 위해 새로운 데이터셋을 생성

Y2_train = Y2_train.rename(columns={'target':'binned_target'})



train_df = pd.merge(X_train, Y_train['target'], how='left', on=['td', 'code'])

train_df = pd.merge(train_df, Y2_train['binned_target'], how='left', on=['td', 'code'])

train_df['binary_target'] = train_df['target'].apply(lambda x: 1 if x >= 0 else 0)



train_df = train_df.reset_index()



train_df['td'] = train_df['td'].str[1:].astype('int')

test_df['td'] = test_df['td'].str[1:].astype('int')
train_df.head()
# 3번 이하 등장하는 code (종목) 제거

temp_dict = Counter(train_df['code'])

outlier_codes = [k for k, v in set(temp_dict.items()) if v <= 3]

train_df = train_df.loc[~train_df['code'].isin(outlier_codes)]



# F feature만 추출

F_cols = [col for col in train_df.columns if col.startswith('F')]



train_df = preprocess_nan(train_df, F_cols)

test_df = preprocess_nan(test_df, F_cols)



# code (종목) 라벨 인코딩, group 통계 파생변수 생성을 위해서

le = LabelEncoder().fit(pd.concat([train_df['code'], test_df['code']]))

for df in [train_df, test_df]:

    df['code_encoded'] = le.transform(df['code'])



# EMA (Exponential Moving Average)를 각각의 F 피쳐에 적용하여 새로운 파생변수를 생성

train_df = train_df.sort_values(by=['code', 'td']).reset_index(drop=True)

test_df = test_df.sort_values(by=['code', 'td']).reset_index(drop=True)

for feat_col in F_cols:

    train_df[f'{feat_col}_EMA_3'] = EMA(train_df, feat_col, 3)

    test_df[f'{feat_col}_EMA_3'] = EMA(test_df, feat_col, 3)



# code (종목)별로 통계 기반 aggregation 파생변수 생성 (mean)

train_df, test_df = code_agg(train_df, test_df, merge_columns=['code'], columns=F_cols, aggs=['mean'])
# column 구분

target_cols = [col for col in train_df.columns if col in ['target', 'binned_target', 'binary_target']]

remove_cols = ['td', 'code']

remove_cols.append('code_encoded') # label encoding 피쳐는 사용하지 않는다.

feat_cols = [col for col in train_df.columns if col not in target_cols+remove_cols]
# check if NaN exists

print("number of NaNs in the Train Dataset: {}".format(train_df[feat_cols].isnull().sum().sum()))

print("number of NaNs in the Test Dataset: {}".format(test_df[feat_cols].isnull().sum().sum()))
# 학습에 사용하는 feature 칼럼들

for col in feat_cols:

    print(col, end=' ')

print("\n\ntotal features used for training: {}".format(len(feat_cols)))
# 전처리된 tree계열과 mlp 모델 학습을 위한 최종 데이터셋 (train, validation split + feature normalizing)

dataset_dict = make_dict(train_df, test_df, feat_cols, 'binary_target')
# Light GBM parameters

lgb_params = {

                'objective':'binary',

                'boosting_type':'gbdt',

                'metric':'auc',

                'n_jobs':-1,

                'learning_rate':0.01,

                'num_leaves': 2**8,

                'max_depth':-1,

                'tree_learner':'serial',

                'colsample_bytree': 0.7,

                'subsample_freq':1,

                'subsample':0.7,

                'n_estimators':3000,

                'max_bin':255,

                'verbose':-1,

                'seed': 42,

                'early_stopping_rounds':100, 

                } 
X_train, X_valid, y_train, y_valid = train_test_split(train_df[feat_cols], train_df[target_cols[2]], test_size=0.1, shuffle=True, random_state=42)



tr_data = lgb.Dataset(X_train, label=y_train)

vl_data = lgb.Dataset(X_valid, label=y_valid) 



clf = lgb.train(

    lgb_params,

    tr_data,

    valid_sets = [tr_data, vl_data],

    verbose_eval = 100 ,

)   



preds = clf.predict(X_valid)

print("\naccuracy score: {:.4f}".format(accuracy_score(y_valid, to_binary(preds))))

print("roc_auc score: {:.4f}".format(roc_auc_score(y_valid, preds)))
# dataset 1: train (1 <= td <= 172) valid (172 < td <= 206)

preds = make_predictions(dataset_dict, train_df, feat_cols, lgb_params, valid_type='hold_out')
# 정확한 구현은 아니지만, 시간 단위가 비식별화 되어있어 나누기가 모호하기 때문에 다음과 같이 진행한다.

# dataset 1: train (1 <= td <= 172) valid (172 < td <= 206)

# dataset 2: train (1 <= td <= 206) valid (206 < td <= 240)

# dataset 3: train (1 <= td <= 240) valid (240 < td <= 276)

preds = make_predictions(dataset_dict, train_df, feat_cols, lgb_params, valid_type='sliding_window')
# check if using cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_cuda = True if device.type == 'cuda' else False

use_cuda
# output path 

output_dir = Path('./', 'output')

output_dir.mkdir(exist_ok=True, parents=True)
# train setting (hyper parameters)

valid_type = 'sliding_window' # in ['hold_out', 'sliding_window']

# 설명을 덧붙이자면, hold_out 방법은 어떤 피쳐가 좋은지 실험해보고자 할 때, 짧은 시간 안에 수행할 수 있기에 이점이 있는 반면에

# sliding_window 방법은 본격적으로 학습을 진행하고자 할 때 사용하도록 한다.

num_epochs = 120

verbose_epoch = 20

lr = 0.00025

batch_size = 1024

num_classes = 1 # 이진 분류

num_feats = len(feat_cols)



criterion = nn.BCELoss()

# criterion = nn.BCEWithLogitsLoss()
# validation 종류에 따라 나눈다 (hold_out, sliding_window 두 종류)

if valid_type == 'hold_out':

    print(f"training starts (hold-out)")

    

    model = build_model(device, 'MLP', num_classes, num_feats)

    optimizer = AdamW(model.parameters(), lr, weight_decay=0.000025)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)



    X_train = dataset_dict['X_train3']

    y_train = dataset_dict['y_train3']

    X_valid = dataset_dict['X_valid3']

    y_valid = dataset_dict['y_valid3']

    

    train_loader = build_dataloader(X_train, y_train, batch_size, shuffle=True)

    valid_loader = build_dataloader(X_valid, y_valid, batch_size, shuffle=False)

    train_mlp_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, use_cuda, verbose_epoch)

    

# total prediction (using 3 models)

elif valid_type == 'sliding_window':

    window_num = 3

    for num in range(1, window_num+1):    

        

        print(f"num {num} dataset training starts (sliding_window)")

        '''

        dataset 1: train (1 <= td <= 172) valid (172 < td <= 206)

        dataset 2: train (1 <= td <= 206) valid (206 < td <= 240)

        dataset 3: train (1 <= td <= 240) valid (240 < td <= 276)

        '''

        model = build_model(device, 'MLP', num_classes, num_feats)

        optimizer = AdamW(model.parameters(), lr, weight_decay=0.000025)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)



        X_train = dataset_dict[f'X_train{num}']

        y_train = dataset_dict[f'y_train{num}']

        X_valid = dataset_dict[f'X_valid{num}']

        y_valid = dataset_dict[f'y_valid{num}']

        train_loader = build_dataloader(X_train, y_train, batch_size, shuffle=True)

        valid_loader = build_dataloader(X_valid, y_valid, batch_size, shuffle=False)

        path = f'best_model_{num}.pt'

        train_mlp_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, use_cuda, verbose_epoch, path)

        del model; gc.collect()

        print("#"*150)

else:

    raise NotImplementedError
# MLP inference [hold_out or sliding_window]

if valid_type == 'hold_out':



    test_loader = build_dataloader(dataset_dict['x_test'], Y=None, batch_size=batch_size, shuffle=False)



    model_path = os.listdir('../working/output')[0]

    model = build_model(device, 'MLP', num_classes, num_feats)

    model.load_state_dict(torch.load(os.path.join('../working/output/', model_path)))

    model.to(device)

    total_preds = mlp_inference(model, test_loader, batch_size, use_cuda)

    total_preds = np.where(total_preds >= 0.5, 1, 0)

    

elif valid_type == 'sliding_window':

    

    model_list = os.listdir('../working/output')

    total_preds = []

    window_num = 3

    

    for i in range(window_num):

        

        batch_size = 1024

        test_loader = build_dataloader(dataset_dict['x_test'], Y=None, batch_size=batch_size, shuffle=False)



        model = build_model(device, 'MLP', num_classes, num_feats)

        model_path = model_list[i]

        model.load_state_dict(torch.load(os.path.join('../working/output/', model_path)))

        model.to(device)



        test_preds = mlp_inference(model, test_loader, batch_size, use_cuda)

        total_preds.append(test_preds)

        

    # logit 단에서 모델 세 개의 결과값을 평균 (3개의 window 기반으로 검증된)

    total_preds = np.mean(total_preds, axis=0)

    total_preds = np.where(total_preds >= 0.5, 1, 0)



else:

    raise NotImplementedError
test_df['prediction'] = total_preds

submission = test_df[['td', 'code', 'prediction']].set_index(['td', 'code'])

submission.to_csv('submission.csv', index=False)
submission['prediction'].value_counts()
# code 'A507' 학습 (40 epoch)

lstm_train_visualize(train_df, 'A507', feat_cols, num_epochs=40, seq_len=12)
# code 'A507' 학습 (100 epoch)

lstm_train_visualize(train_df, 'A507', feat_cols, num_epochs=100, seq_len=12)
# code 'A507' 학습 (300 epoch)

lstm_train_visualize(train_df, 'A507', feat_cols, num_epochs=300, seq_len=12)