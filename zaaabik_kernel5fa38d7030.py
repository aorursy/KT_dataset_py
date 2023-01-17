import numpy as np

import pandas as pd

from zipfile import ZipFile

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, freqz

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, roc_auc_score, euclidean_distances

from catboost import CatBoostClassifier

from scipy.signal import decimate, freqz
def get_subject_data_df(zip_file, subject_id, series_ids, folder):

    df = pd.DataFrame()

    for series_id in series_ids:

        file_name = f'{folder}/subj{subject_id}_series{series_id}_data.csv'

        unzipped_file = zip_file.open(file_name)

        tmp_df = pd.read_csv(unzipped_file, index_col='id')

        df = pd.concat([df, tmp_df])

    return df



def get_subject_target_df(zip_file, subject_id):

    df = pd.DataFrame()

    for series_id in range(1,9):

        file_name = f'train/subj{subject_id}_series{series_id}_events.csv'

        unzipped_file = zip_file.open(file_name)

        tmp_df = pd.read_csv(unzipped_file, index_col='id')

        df = pd.concat([df, tmp_df])

    return df
def butter_lowpass(cutoff, fs, order=5):

    nyq = 0.5 * fs

    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    return b, a



def butter_lowpass_filter(data, cutoff, fs, order=5):

    b, a = butter_lowpass(cutoff, fs, order=order)

    freqz(b,a)

    y = lfilter(b, a, data, axis=0)

    return y
def butter_highpass(cutoff, fs, order=5):

    nyq = 0.5 * fs

    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    return b, a



def butter_highpass_filter(data, cutoff, fs, order=5):

    b, a = butter_highpass(cutoff, fs, order=order)

    y = lfilter(b, a, data, axis=0)

    return y
def preprocess_data(x, downsample_rate):    

    signal = decimate(x, downsample_rate, axis=0)

    return signal
subject_ids = range(1,13)

# subject_ids = subject_ids[:1]

train_zip_path = '/kaggle/input/grasp-and-lift-eeg-detection/train.zip'

test_zip_path = '/kaggle/input/grasp-and-lift-eeg-detection/test.zip'

train_zip = ZipFile(train_zip_path)

test_zip = ZipFile(test_zip_path)



submission = []

idx = []

droprate = 50

fs = 500

low_cut_off = 0.2

high_cut_off = 4

auc = []



for subject_id in subject_ids:

    x_train = get_subject_data_df(train_zip, subject_id, range(1, 9), 'train')

    x_test = get_subject_data_df(test_zip, subject_id, [9,10], 'test')

    target_df = get_subject_target_df(train_zip, subject_id)

    idx.append(np.array(x_test.index))



    m,n = x_test.shape

    pred = np.zeros((m, 6))

    scaler = StandardScaler()

    x_train = butter_lowpass_filter(x_train, low_cut_off, fs)

    x_train = butter_highpass_filter(x_train, high_cut_off, fs)

    scaler.fit(x_train)    

    x = x_train[::droprate]

    x = scaler.transform(x)

    x_test = butter_lowpass_filter(x_test, low_cut_off, fs)

    x_test = butter_highpass_filter(x_test, high_cut_off, fs)

    x_test = scaler.transform(x_test)

    

    cb = LogisticRegression(solver='lbfgs', max_iter=200)

    

    for i in range(6):

        y = target_df[::droprate].values[:, i]

        score = cross_val_score(cb, x, y, cv=2, scoring='roc_auc').mean()

        auc.append(score)

        cb.fit(x,y)

        pred[:,i] = cb.predict_proba(x_test)[:, 1]

    submission.append(pred)

    print('end of subject: ', subject_id)
auc
np.mean(auc)
submission_concat = np.concatenate(submission, axis=0)

idx_concat = np.concatenate(idx)
cols = ['HandStart','FirstDigitTouch',

        'BothStartLoadPhase','LiftOff',

        'Replace','BothReleased']

submission_csv = pd.DataFrame(index=idx_concat, columns=cols, data=submission_concat)
submission_csv.to_csv('submission_classes.csv.gz', index_label='id', float_format='%.5f',  compression='gzip')
# import torch.nn as nn

# import torch.nn.functional as F
# class CNN(nn.Module):

#     def __init__(self):

#         super(CNN, self).__init__()

#         self.conv1 = nn.Conv2d(3, 6, 5)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(6, 16, 5)

#         self.fc1 = nn.Linear(16 * 5 * 5, 120)

#         self.fc2 = nn.Linear(120, 84)

#         self.fc3 = nn.Linear(84, 6)



#     def forward(self, x):

#         x = self.pool(F.relu(self.conv1(x)))

#         x = self.pool(F.relu(self.conv2(x)))

#         x = x.view(-1, 16 * 5 * 5)

#         x = F.relu(self.fc1(x))

#         x = F.relu(self.fc2(x))

#         x = self.fc3(x)

#         x = F.sigmoid(x)

#         return x
# net = CNN()
# torch_singular_values = torch.Tensor(singular_values)

# torch_mean_targets = torch.Tensor(mean_targets)



# optimizer = optim.Adam(net.parameters())

# loss_function = nn.BCEWithLogitsLoss()

# batch_size = 32

# batch_count = batches // batch_size



# output = net.forward(torch_singular_values)

# loss = loss_function(output, torch_mean_targets)

# print(loss)



# net.train()

# for epoch in range(40):

#     for i in range(batch_count):

#         batch = torch_singular_values[i * batch_size: (i + 1) * batch_size]

#         target = torch_mean_targets[i * batch_size: (i + 1) * batch_size]

#         optimizer.zero_grad()

#         output = net.forward(batch)

#         loss = loss_function(output, target)    

#         loss.backward()



#         optimizer.step()



#         output = net.forward(torch_singular_values)

#         loss = loss_function(output, torch_mean_targets)

#         print(loss.data)
# window_size = 1000

# n,m = data_df.shape

# batches = n // window_size + 1

# singular_values = np.zeros((batches, m))

# for i in range(0, batches):

#     window = data_df[i * window_size : (i + 1) * window_size]

#     u,s,vh = np.linalg.svd(window, False)

#     singular_values[i] = s
# mean_targets = np.zeros((batches, 6))

# for i in range(0, batches):

#     target_window = target_df[i * window_size : (i + 1) * window_size]

#     mean_target = np.mean(target_window)     

#     mean_targets[i] = mean_target
# import torch

# import torch.nn as nn

# import torch.nn.functional as F

# import torch.optim as optim
# class Net(nn.Module):

#     def __init__(self, in_size):

#         super(Net, self).__init__()

#         self.lin_layer1 = nn.Linear(in_size, 100)

#         self.lin_layer2 = nn.Linear(100, 100)

#         self.lin_layer3 = nn.Linear(100, 6)

#         self.Act = nn.ReLU()

#         self.Sigm = nn.Sigmoid()



#     def forward(self, x):

#         x = self.lin_layer1(x)

#         x = self.Act(x)

#         x = self.lin_layer2(x)

#         x = self.Act(x)

#         x = self.lin_layer3(x)

#         # x = self.Sigm(x)

#         return x
# m, n = singular_values.shape

# net = Net(n)
# torch_singular_values = torch.Tensor(singular_values)

# torch_mean_targets = torch.Tensor(mean_targets)



# optimizer = optim.Adam(net.parameters())

# loss_function = nn.BCEWithLogitsLoss()

# batch_size = 32

# batch_count = batches // batch_size



# output = net.forward(torch_singular_values)

# loss = loss_function(output, torch_mean_targets)

# print(loss)



# net.train()

# for epoch in range(40):

#     for i in range(batch_count):

#         batch = torch_singular_values[i * batch_size: (i + 1) * batch_size]

#         target = torch_mean_targets[i * batch_size: (i + 1) * batch_size]

#         optimizer.zero_grad()

#         output = net.forward(batch)

#         loss = loss_function(output, target)    

#         loss.backward()



#         optimizer.step()



#         output = net.forward(torch_singular_values)

#         loss = loss_function(output, torch_mean_targets)

#         print(loss.data)

    
# def get_subject_test_data_df(zip_file, subject_id):

#     df = pd.DataFrame()

#     for series_id in range(9, 11):

#         file_name = f'test/subj{subject_id}_series{series_id}_data.csv'

#         unzipped_file = zip_file.open(file_name)

#         tmp_df = pd.read_csv(unzipped_file, index_col='id')

#         df = pd.concat([df, tmp_df])

#     return df
# test_zip_path = '/kaggle/input/grasp-and-lift-eeg-detection/test.zip'

# test_zip = ZipFile(test_zip_path)

# for subject_id in subject_ids:

#     test_data_df = get_subject_test_data_df(test_zip, subject_id)
# n,m = test_data_df.shape

# batches = n // window_size + 1

# test_singular_values = np.zeros((batches, m))

# for i in range(0, batches):

#     window = test_data_df[i * window_size : (i + 1) * window_size]

#     u,s,vh = np.linalg.svd(window, False)

#     test_singular_values[i] = s
# torch_test_singular_values = torch.Tensor(test_singular_values)

# test_output = net.forward(torch_test_singular_values)

# test_output = nn.functional.sigmoid(test_output)
# sample_submission_path = '/kaggle/input/grasp-and-lift-eeg-detection/sample_submission.csv.zip'

# submission = pd.read_csv(sample_submission_path, index_col='id')
# out = np.array(test_output.data)

# out = out.repeat(1000, axis=0)

# m,n = out.shape

# submission.iloc[:m, :] = out
# submission.to_csv('result2.csv')