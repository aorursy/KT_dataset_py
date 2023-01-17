# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/reducing-commercial-aviation-fatalities/sample_submission.csv")

test = pd.read_csv("../input/reducing-commercial-aviation-fatalities/test.csv")

train = pd.read_csv("../input/reducing-commercial-aviation-fatalities/train.csv")
test.head()
train.head()
train_time = train['time']

test_time = test['time']
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(15,10))

sns.distplot(train_time,label="train time")

sns.distplot(test_time,label="test time")

plt.legend()

plt.xlabel("Time (s)")

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(train['ecg'],label="train ecg",hist=False)

sns.distplot(test['ecg'],label="test ecg",hist=False)

plt.legend()

plt.xlabel("ECG")

plt.figure()
train_A = train[train['event'] == 'A']

train_B = train[train['event'] == 'B']

train_C = train[train['event'] == 'C']

train_D = train[train['event'] == 'D']
plt.figure(figsize=(15,10))

sns.distplot(train_A['ecg'],label="train_A ecg",hist=False)

sns.distplot(train_B['ecg'],label="train_B ecg",hist=False)

sns.distplot(train_C['ecg'],label="train_C ecg",hist=False)

sns.distplot(train_D['ecg'],label="train_D ecg",hist=False)

plt.legend()

plt.xlabel("ECG")

plt.figure()
plt.figure(figsize=(15,10))

sns.distplot(train_A['gsr'],label="train_A gsr")

sns.distplot(train_B['gsr'],label="train_B gsr")

sns.distplot(train_C['gsr'],label="train_C gsr")

sns.distplot(train_D['gsr'],label="train_D gsr")

plt.legend()

plt.xlabel("GSR")

plt.figure()
plt.figure(figsize=(15,10))

sns.distplot(train_A['r'],label="train_A r")

sns.distplot(train_B['r'],label="train_B r")

sns.distplot(train_C['r'],label="train_C r")

sns.distplot(train_D['r'],label="train_D r")

plt.legend()

plt.xlabel("Respiration")

plt.figure()
plt.figure(figsize=(15,10))

sns.distplot(train['ecg'],label="train ECG")

sns.distplot(test['ecg'],label="test ECG")

plt.legend()

plt.xlabel("ECG")

plt.figure()
eeg_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]

k=0



plt.figure(figsize=(20,25))

for i in eeg_features:

    k+=1

    plt.subplot(5,5,k)

    sns.distplot(train.sample(10000)[i],label="train "+i,hist=False)

    sns.distplot(test.sample(10000)[i],label="test "+i,hist=False)

    plt.xlim((-500, 500))

    plt.legend()

    

plt.show()

    

    
train.count()
test.count()