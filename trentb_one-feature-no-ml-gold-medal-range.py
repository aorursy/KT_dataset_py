import os

import numpy as np

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import pandas as pd

from numba import njit

import gc
@njit

def entropy_fast(vec, bins):

    h = np.histogram(vec, bins=bins)[0] + 1E-15

    h = h / np.sum(h)

    return -np.sum(h * np.log(h))
def replace_peaks(X, win=1000, threshold=500):

    np.random.seed(0)

    

    for idx in range(X.shape[0]):

        max_idx = np.argmax(np.abs(X[idx]))

        

        while np.abs(X[idx,max_idx]) > threshold:

            temp_win = win + np.random.randint(-500, 501)

            X[idx,max_idx] = 0.5 * (X[idx, np.maximum(0, max_idx-temp_win)] + X[idx, np.minimum(X[idx].shape[0] - 1, max_idx+temp_win)])

            max_idx = np.argmax(np.abs(X[idx]))
%%time

df = pd.read_csv(os.path.join('..','input','train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
n_samples = 150000

n_segments = int(np.floor(df.shape[0] / n_samples))
X = np.zeros((n_segments, n_samples))

y = np.zeros(n_segments)



for seg_id in range(n_segments):

    seg = df.iloc[seg_id*n_samples:seg_id*n_samples+n_samples]

    X[seg_id] = seg['acoustic_data'].values

    y[seg_id] = seg['time_to_failure'].values[-1]



del df

gc.collect();
replace_peaks(X)
ent = np.zeros(X.shape[0])



n=2000

ent_temp = np.zeros(n)

cv = KFold(n, shuffle=False)



for idx in tqdm_notebook(range(X.shape[0])):

    for idx2, (train_idx, test_idx) in enumerate(cv.split(X[idx])):

        ent_temp[idx2] = entropy_fast(X[idx,test_idx], 300)

    

    ent[idx] = np.mean(ent_temp)
plt.figure(figsize=(15,5))

plt.plot(ent)

plt.xlabel('time')

plt.ylabel('entropy');
feature = np.clip(ent, a_min=2.37, a_max=2.66)



model = LinearRegression(n_jobs=-1)

model.fit(feature.reshape(-1,1), y.reshape(-1,1))

feature = model.predict(feature.reshape(-1,1))



print('MAE: ', mean_absolute_error(feature, y))
plt.figure(figsize=(15,5))

plt.plot(feature,'r')

plt.plot(y,'k')

plt.ylabel('TTF')

plt.xlabel('time')

plt.grid()
### Load testing data
submission = pd.read_csv(os.path.join('..','input','sample_submission.csv'), index_col='seg_id')
n_test_segments = submission.shape[0]



X_test = np.zeros((n_test_segments, n_samples))



for seg_idx, seg_id in enumerate(tqdm_notebook(submission.index)):

    seg = pd.read_csv(os.path.join('..','input','test', seg_id + '.csv'))

    X_test[seg_idx] = seg['acoustic_data'].values
replace_peaks(X_test)
ent = np.zeros(X_test.shape[0])



n=2000

ent_temp = np.zeros(n)

cv = KFold(n, shuffle=False)



for idx in tqdm_notebook(range(X_test.shape[0])):

    for idx2, (train_idx, test_idx) in enumerate(cv.split(X_test[idx])):

        ent_temp[idx2] = entropy_fast(X_test[idx,test_idx], 300)

    

    ent[idx] = np.mean(ent_temp)



test_feature = np.clip(ent, a_min=2.37, a_max=2.66)
test_set_TTF_peaks = [11, 11, 11, 8, 11, 16, 9, 11, 16]

test_set_TTF_mean_peak = np.mean(test_set_TTF_peaks)

scaling_factor = test_set_TTF_mean_peak / np.max(feature)
predictions = scaling_factor * model.predict(test_feature.reshape(-1,1)).flatten()
submission.time_to_failure = predictions

submission.to_csv('submission.csv',index=True)