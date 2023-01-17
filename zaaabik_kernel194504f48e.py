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

from scipy.signal import decimate, fftconvolve

from numba import jit
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
def butter_highpass(cutoff, fs, order=5):

    nyq = 0.5 * fs

    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)

    return b, a



def butter_highpass_filter(data, cutoff, fs, order=5):

    b, a = butter_highpass(cutoff, fs, order=order)

    y = lfilter(b, a, data, axis=0)

    return y
def butter_lowpass(cutoff, fs, order=5):

    nyq = 0.5 * fs

    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    return b, a



def butter_lowpass_filter(data, cutoff, fs, order=5):

    b, a = butter_lowpass(cutoff, fs, order=order)

    freqz(b,a)

    y = lfilter(b, a, data, axis=0)

    return y
def preprocess_data(x, low_cut_off, high_cut_off, fs):

    x = butter_lowpass_filter(x, low_cut_off, fs)

    x = butter_highpass_filter(x, high_cut_off, fs)

    return x
subject_ids = range(1,13)

subject_ids = subject_ids[:1]

train_zip_path = '/kaggle/input/grasp-and-lift-eeg-detection/train.zip'

test_zip_path = '/kaggle/input/grasp-and-lift-eeg-detection/test.zip'

train_zip = ZipFile(train_zip_path)

test_zip = ZipFile(test_zip_path)

cb = LogisticRegression(solver='lbfgs', max_iter=400)
### PREDICTION BASED ON rank-r appoximation of the sliding window ###

subject_ids = range(1,13)

subject_ids = [1]

aucs_sl = []

droprate = 100

h = 16

l = 0.2



def get_sliding_window (i, length, data):

    return data[i-length:i+1,:]



for subject_id in subject_ids:

    

    x_train = get_subject_data_df(train_zip, subject_id, range(1, 9), 'train')

    target_df = get_subject_target_df(train_zip, subject_id)

    x_train = preprocess_data(x_train, l, h, 500)

    

    r = 32

    

    sliding_window = [50,100,200,500]



    

    for slide_l in sliding_window:

        

        auc = []

        

        start = max(slide_l, droprate)

        indices = list(range(x_train.shape[0]))[start::droprate]

        x = np.zeros((len(indices),(slide_l+1)*32))

        

        for i in range (len(indices)):

            window = get_sliding_window (indices[i], slide_l, x_train)

            u,s,vh = np.linalg.svd(window, False)

            u = u[:,:r]

            s = s[:r]

            vh = vh[:r, :]

            new = u @ np.diag(s) @ vh

            x [i,:] = new.flatten()



        scaler = StandardScaler()

        scaler.fit(x)

        x = scaler.transform(x)



        print(f'done with training set {slide_l}')



        for i in range(6):

            y = target_df[start::droprate].values[:, i]

            score = cross_val_score(cb, x, y, cv=2, scoring='roc_auc').mean()

            auc.append(score)



            # print ('done for model', i)

            

        aucs_sl.append((slide_l, np.array(auc).mean()))
aucs_sl_np = np.array(aucs_sl)

df = pd.DataFrame(data=aucs_sl_np, columns=['window size', 'roc auc'])

df.to_csv('r32_best_filters.csv')
df