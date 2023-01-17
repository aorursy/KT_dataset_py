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
!git clone https://cortexbci.admin@bitbucket.org/cortexbci.admin/eeg_artefact_dataset.git
!cd eeg_artefact_dataset && ls
train = pd.read_csv('./eeg_artefact_dataset/S02_21.08.20_14.34.32.csv')
train.head()
classes = train['MarkerValueInt']
from scipy.signal import butter, lfilter

'''
Code for Butterworth Bandpass Filters
'''
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
train = train[1:]
train[train['MarkerType']==1]
del train['MarkerValueInt']
del train['EEG.Counter']
del train['Timestamp']
del train['MarkerIndex']
del train['MarkerType']
orig_data = train.values
l = 2
r = 12
train['1'] = butter_bandpass_filter(train['EEG.AF3'].values,l,r,128,2)
train['2'] = butter_bandpass_filter(train['EEG.F7'].values,l,r,128,2)
train['3'] = butter_bandpass_filter(train['EEG.F3'].values,l,r,128,2)
train['4'] = butter_bandpass_filter(train['EEG.FC5'].values,l,r,128,2)
train['5'] = butter_bandpass_filter(train['EEG.T7'].values,l,r,128,2)
train['6'] = butter_bandpass_filter(train['EEG.P7'].values,l,r,128,2)
train['7'] = butter_bandpass_filter(train['EEG.O1'].values,l,r,128,2)
train['8'] = butter_bandpass_filter(train['EEG.O2'].values,l,r,128,2)
train['9'] = butter_bandpass_filter(train['EEG.P8'].values,l,r,128,2)
train['10'] = butter_bandpass_filter(train['EEG.T8'].values,l,r,128,2)
train['11'] = butter_bandpass_filter(train['EEG.FC6'].values,l,r,128,2)
train['12'] = butter_bandpass_filter(train['EEG.F4'].values,l,r,128,2)
train['13'] = butter_bandpass_filter(train['EEG.F8'].values,l,r,128,2)
train['14'] = butter_bandpass_filter(train['EEG.AF4'].values,l,r,128,2)
train['1'] = train['EEG.AF3']
train['2'] = train['EEG.F7']
train['3'] = train['EEG.F3']
train['4'] = train['EEG.FC5']
train['5'] = train['EEG.T7']
train['6'] = train['EEG.P7']
train['7'] = train['EEG.O1']
train['8'] = train['EEG.O2']
train['9'] = train['EEG.P8']
train['10'] = train['EEG.T8']
train['11'] = train['EEG.FC6']
train['12'] = train['EEG.F4']
train['13'] = train['EEG.F8']
train['14'] = train['EEG.AF4']
del train['EEG.AF3']
del train['EEG.F7']
del train['EEG.F3']
del train['EEG.FC5']
del train['EEG.T7']
del train['EEG.P7']
del train['EEG.O1']
del train['EEG.O2']
del train['EEG.P8']
del train['EEG.T8']
del train['EEG.FC6']
del train['EEG.F4']
del train['EEG.F8']
del train['EEG.AF4']
import matplotlib.pyplot as plt
import scipy.signal as sps
from sklearn.decomposition import FastICA
train_nparr = train.values
ica = FastICA(n_components=14)
ica.fit(train_nparr)
components = ica.transform(train_nparr)
components.shape
s = orig_data*train_nparr
true_artifacts = train[train['MarkerType']==1].index
true_artifacts
n = 13
plt.subplot(3, 1, 2)
plt.plot([[np.nan, np.nan, np.nan]])  # advance the color cycler to give the components a different color :)
# plt.plot(components + [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
plt.plot(s[6300:6500,n:n+1])
# plt.plot(components + [0.5, 1.0, 1.5])
# plt.yticks([0.5, 1.0, 1.5], ['0', '1', '2'])
plt.ylabel('components')
i = 6000
count = 0
arr = []
while i < 46065:
    if s[i,n:n+1] > 18500000: #and s[i-1,n:n+1] <= s[i,n:n+1] and s[i+1,n:n+1] >= s[i,n:n+1]:
        i += 64 #Window Size
        count += 1
        arr.append(i)
    i += 1
i = 0
j = 0
diff = []
miss_artifact = []
miss_predict = []
while i < len(arr) and j < len(true_artifacts):
    if  arr[i] - true_artifacts[j] > 100:
        miss_artifact.append(true_artifacts[j])
        j += 1
    elif arr[i] - true_artifacts[j] < -100:
        miss_predict.append(arr[i])
        i += 1
    else:
        diff.append(arr[i] - true_artifacts[j])
        i += 1
        j += 1
print('Artifacts missed:',len(miss_artifact),miss_artifact)
print('Artifacts mispredicted:',len(miss_predict),miss_predict)
print('General Diffrence between values:',diff)
print('Number of artifacts predicted:',count)
print('Predicted Values:',arr)
print('True Values:',true_artifacts)
restored = ica.inverse_transform(components)
train_nparr[:,:1]
n = 0
val = restored[:,n:n+1]-train_nparr[:,n:n+1]
plt.subplot(3, 1, 2)
plt.plot([[np.nan, np.nan, np.nan]])  # advance the color cycler to give the components a different color
plt.plot(val[6700:6800])
# plt.plot(components + [0.5, 1.0, 1.5])
# plt.yticks([0.5, 1.0, 1.5], ['0', '1', '2'])
plt.ylabel('components')
print(val[6440],val[6719],val[7116],val[7363],val[7774])
count = 0
for i in range(len(classes)):
    if classes[i] == 22 or classes[i] == 23:
        val[i] += (10*(10**-13))
    else:
        count += 1
plt.plot(val[00:7400])
