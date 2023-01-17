# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import lognorm, gamma
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

%pip install saxpy
import saxpy as sax

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
# %pip install stumpy
# import stumpy

from numba import cuda

# %pip install matrixprofile
# import matrixprofile as mp



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sample_submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
train.shape
test.shape
train.head()
test.head()
f, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), sharex=True)
sns.distplot(train.signal, ax=axes[0])
sns.distplot(test.signal, color="g", ax=axes[1])
plt.show()
plt.hist(train.open_channels)
plt.show()
fig, axs = plt.subplots(2, figsize=(15, 10))
fig.suptitle('signal')
axs[0].plot(train.time, train.signal)
axs[1].plot(test.time, test.signal, color="g")
plt.show()
fig, axs = plt.subplots(2, figsize=(15, 10))
fig.suptitle('signal')
axs[0].plot(train[:500000].time, train[:500000].signal)
axs[1].plot(train[4500000:5000000].time, train[4500000:5000000].signal, color="g")
plt.show()
# # MATRIX PROFILE 

# signal_train = train[4500000:5000000].signal.values
# window_size = 50  
# all_gpu_devices = [device.id for device in cuda.list_devices()]

# matrix_profile = stumpy.gpu_stump(signal_train, m=window_size, device_id=all_gpu_devices)

# signal_train = train[:500000].signal.values
# signal_test = test[:500000].signal.values
# # window = 8
# # profile = mp.compute(signal, windows=window)
# # profile = mp.discover.motifs(profile, k=1)
# # figures = mp.visualize(profile)
# profile, figures = mp.analyze(signal_train)
plt.hist(train.query('open_channels == "0"').signal, alpha=0.5, label='channel_0')
plt.hist(train.query('open_channels == "1"').signal, alpha=0.5, label='channel_1')
plt.hist(train.query('open_channels == "2"').signal, alpha=0.5, label='channel_2')
plt.hist(train.query('open_channels == "3"').signal, alpha=0.5, label='channel_3')
plt.legend(loc='upper right')
plt.show()
class_train = train.groupby(train.open_channels).size()
for i in range(11):
    print("class_{0} : {1} : {2:.2f} %".format(i, class_train[i], class_train[i]/len(train)*100))
# SAX transform
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.sax import sax_via_window

# sax_train = sax_via_window(train.signal.values, win_size=6, paa_size=3, alphabet_size=3, nr_strategy=None)
# sax_test = sax_via_window(test.signal.values, win_size=6, paa_size=3, alphabet_size=3, nr_strategy=None)

# sax_train_df = pd.DataFrame(sax_train.items(), columns=['seq', 'index'])
# sax_test_df = pd.DataFrame(sax_test.items(), columns=['seq', 'index'])

# sax_train_df = sax_train_df.sort_values(by=['seq']).query('seq != "aaa"')
# sax_test_df = sax_test_df.sort_values(by=['seq']).query('seq != "ccc"')
train['rolling_5_mean'] = train.signal.rolling(5).mean()
test['rolling_5_mean'] = test.signal.rolling(5).mean()

train['rolling_10_mean'] = train.signal.rolling(10).mean()
test['rolling_10_mean'] = test.signal.rolling(10).mean()

train['rolling_20_mean'] = train.signal.rolling(20).mean()
test['rolling_20_mean'] = test.signal.rolling(20).mean()

train['rolling_50_mean'] = train.signal.rolling(50).mean()
test['rolling_50_mean'] = test.signal.rolling(50).mean()
plt.figure(figsize=(8, 8))
sns.heatmap(train.corr(), annot=True)
X_train = train.drop(['time','open_channels'], axis=1)
y_train = train['open_channels'].values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 890)

X_test = test
X_test = X_test.drop(['time'], axis=1)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=5, random_state=3456, shuffle=True)

# parameters = {'estimator__n_estimators': [500, 1000, 1500, 7000, 9000, 12000], 'estimator__learning_rate': [0.001, 0.01, 0.03, 0.05]}
parameters = {'n_estimators': 1000, 'learning_rate': 0.05}
clf = XGBClassifier(tree_method='gpu_hist')
clf.set_params(**parameters)
clf
xgb_classifier = OneVsRestClassifier(clf)

# model_tunning = GridSearchCV(xgb_classifier, param_grid=parameters, scoring='f1_macro', verbose=10, cv=skf)

# model_tunning.fit(X_train, y_train)

# print(model_tunning.best_score_)
# print(model_tunning.best_params_)
xgb_classifier.fit(X_tr, y_tr)
y_pred = xgb_classifier.predict(X_val)

# plot predictions
plt.hist(y_pred, alpha=0.5, label='X_val_preds')
plt.legend(loc='upper right')
plt.show()

f1_score(y_pred, y_val, average='macro')
cm = multilabel_confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
f, axes = plt.subplots(3, 5, figsize=(10, 10), sharex=True)
f.tight_layout(pad=2.0)
sns.heatmap(cm[0], annot=True, fmt='d', ax=axes[0, 0], cbar=None)
sns.heatmap(cm[1], annot=True, fmt='d', ax=axes[0, 1], cbar=None)
sns.heatmap(cm[2], annot=True, fmt='d', ax=axes[0, 2], cbar=None)
sns.heatmap(cm[3], annot=True, fmt='d', ax=axes[0, 3], cbar=None)
sns.heatmap(cm[4], annot=True, fmt='d', ax=axes[0, 4], cbar=None)
sns.heatmap(cm[5], annot=True, fmt='d', ax=axes[1, 0], cbar=None)
sns.heatmap(cm[6], annot=True, fmt='d', ax=axes[1, 1], cbar=None)
sns.heatmap(cm[7], annot=True, fmt='d', ax=axes[1, 2], cbar=None)
sns.heatmap(cm[8], annot=True, fmt='d', ax=axes[1, 3], cbar=None)
sns.heatmap(cm[9], annot=True, fmt='d', ax=axes[1, 4], cbar=None)
sns.heatmap(cm[10], annot=True, fmt='d', ax=axes[2, 0], cbar=None)
print(classification_report(y_val, y_pred))
predictions = xgb_classifier.predict(X_test)

# Writing output to file
subm = pd.DataFrame()
subm['time'] = test['time']
subm['open_channels'] = predictions

subm.to_csv("/kaggle/working/" + 'submission.csv', float_format='%.4f', index=False) 
# plot predictions
plt.hist(train.open_channels, alpha=0.5, label='y_train')
plt.hist(predictions, alpha=0.7, label='preds')
plt.legend(loc='upper right')
plt.show()