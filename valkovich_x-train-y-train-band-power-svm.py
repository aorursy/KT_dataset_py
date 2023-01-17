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
import datetime

import numpy as np

import scipy as sp

import scipy.fftpack

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/EEG_prec.txt', sep=' ')
df.shape
df.head()
dti = pd.date_range('2020-02-02', periods=540, freq='2 ms')
df.index = dti
df = df.drop(columns="Unnamed: 21")
df.head()
DD = df.to_numpy()
DD[5]
DD.shape
L = 5*500

K = 12*15

N = 21
XX = np.empty((0, L*21))
for i in range(K):

    XM = DD[i*L:i*L+L,:] ### Выбираем короткие интервалы (длина L) для каждого канала.

    MN = np.ravel(XM.T)  ### Объединяем в одну строку результаты для каждого канала

    XX = np.append(XX, [MN], axis=0) ### Составляем строки в матрице X_train
XX.shape
YY = np.random.randint(3, size=K)
SR = 500            # Sampling rate

ST = 10             # Time interval (sec)

L = ST*SR           # NumPy time interval 

K = int((60/ST)*18) # Number of time intervals to separate

N = 21              # Number of channels
fftfreq = sp.fftpack.fftfreq(L, 1 / 500 ) ### len(PF) = L !!!

i_f = fftfreq > 0  

FM = len(fftfreq[i_f])
df = np.min(fftfreq[i_f]) #

fftfreq_d = fftfreq[0:int(4/df)]           # Delta

fftfreq_t = fftfreq[int(4/df):int(7.5/df)] # Theta

fftfreq_a = fftfreq[int(7.5/df):int(12/df)]# Alpha

fftfreq_b = fftfreq[int(12/df):int(30/df)] # Beta

fftfreq_g = fftfreq[int(30/df):int(80/df)] # Beta
fftfreq_a 
range(200)
df = np.min(fftfreq[i_f]) #

fftfreq_d = range(500)[0:int(4/df)]           # Delta

fftfreq_t = range(500)[int(4/df):int(7.5/df)] # Theta

fftfreq_a = range(500)[int(7.5/df):int(12/df)]# Alpha

fftfreq_b = range(500)[int(12/df):int(30/df)] # Beta

fftfreq_g = range(500)[int(30/df):int(80/df)] # Beta
(fftfreq_d , fftfreq_t ,fftfreq_a, fftfreq_b, fftfreq_g)
fftfreq[fftfreq_d]
XX = np.empty((0, FM*N))

for i in range(K):

    DDA =  np.empty((0, FM))

    for j in range(N):

        DDL = DD[i*L:i*L+L,j]          

        DDL_FFT =  sp.fftpack.fft(DDL) ### Выполняем FFT для каждого короткого интервала и для каждого канала. 

        PF = np.abs(DDL_FFT) ** 2      ### Power spectrum

        PF_P = PF[i_f]                                        ### Getting mean power for each freq band

        DDA = np.append(DDA, [PF_P], axis=0)  ### merge with previously accumulated 

    X_L = np.ravel(DDA)                ### Объединяем в одну строку результаты для всех каналов.

    XX = np.append(XX, [X_L], axis=0)  ### Составляем строки в матрице X_train. 
XX.shape
NB = 5 ### Number of bands

XX = np.empty((0, NB*N))  ### Number of bands X Number of Channels

for i in range(K):

    DDA =  np.empty((0, NB))  ### Number of bands

    for j in range(N):

        DDL = DD[i*L:i*L+L,j]          

        DDL_FFT =  sp.fftpack.fft(DDL) ### Выполняем FFT для каждого короткого интервала и для каждого канала. 

        PF = np.abs(DDL_FFT) ** 2      ### Power spectrum

        PF_P = [np.mean(PF[fftfreq_d]),

                np.mean(PF[fftfreq_t]),

                np.mean(PF[fftfreq_a]),

                np.mean(PF[fftfreq_b]),

                np.mean(PF[fftfreq_g])]       ### Getting mean power for each freq band 

        DDA = np.append(DDA, [PF_P], axis=0)  ### merge with previously accumulated 

    X_L = np.ravel(DDA)                       ### Объединяем в одну строку результаты для всех каналов.

    XX = np.append(XX, [X_L], axis=0)         ### Составляем строки в матрице X_train.  
XX.shape
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size = 0.2)
from sklearn.svm import SVC

clf = SVC()

clf.fit(X_train, y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_test, y_test) * 100, 2)

print (str(acc_svc) + '%')