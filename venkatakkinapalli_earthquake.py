# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
PATH="../input/"

os.listdir(PATH)
print("there are {} number of files in Test folder".format(len(os.listdir(os.path.join(PATH,'test')))))
%%time

train_df = pd.read_csv(os.path.join(PATH,'train.csv'),nrows = 6000000,dtype={

    'acoustic_data': np.int16, 'time_to_failure': np.float32})
train_df.head()
train_df.head(10)

from matplotlib import pyplot as plt
x = train_df['acoustic_data'].values[::100]

y = train_df['time_to_failure'].values[::100]
fig,ax1 = plt.subplots(figsize=(12, 8))

plt.plot(x,color = 'r')

ax1.set_ylabel('Accoustic_data', color = 'r')

ax1.legend(['Accoustic_data'],loc = (0,0.9))

ax2 = ax1.twinx()

plt.plot(y,color = 'b')

plt.grid(True)
def gen_features(X):

    strain = []

    #strain.append(X.mean())

    strain.append(X.std())

    strain.append(X.min())

    strain.append(X.max())

    #strain.append(X.kurtosis())

    #strain.append(X.skew())

    #strain.append(np.quantile(X,0.01))

    #strain.append(np.quantile(X,0.05))

    #strain.append(np.quantile(X,0.95))

    #strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    #strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)
train = pd.read_csv(os.path.join(PATH,'train.csv'), iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})



X_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = gen_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
#machine learning

from catboost import CatBoostRegressor, Pool

#data scaling

from sklearn.preprocessing import StandardScaler

#hyperparameter optimization

from sklearn.model_selection import GridSearchCV

#support vector machine model

from sklearn.svm import NuSVR, SVR

#kernel ridge model

from sklearn.kernel_ridge import KernelRidge
train_pool = Pool(X_train,y_train)

m= CatBoostRegressor(iterations=10000,loss_function='MAE',boosting_type='Ordered')

m.fit(X_train,y_train,silent = True)

m.best_score_
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import NuSVR, SVR





scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)



#parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

#               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]

               #'nu': [0.75, 0.8, 0.85, 0.9, 0.95, 0.97]}]



#reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

#reg1.fit(X_train_scaled, y_train.values.flatten())

#y_pred1 = reg1.predict(X_train_scaled)



#print("Best CV score: {:.4f}".format(reg1.best_score_))

#print(reg1.best_params_)
m2 = SVR(kernel = 'rbf', tol = 0.01, C= 2, gamma=0.02)

m2.fit(X_train_scaled, y_train.values.flatten()) 
PATH1 = os.path.join(PATH,'test')
X_test = pd.DataFrame()

y_test = pd.Series()

for file in os.listdir(os.path.join(PATH,'test')):

    data = pd.read_csv(os.path.join(PATH1,file), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

    data_fet = gen_features(data['acoustic_data'])

    X_test = X_test.append(data_fet,ignore_index = True)

    y_test = y_test.append(pd.Series(os.path.splitext(file)[0]))
X_test_scaled = scaler.transform(X_test)

y_pred_svr = m2.predict(X_test_scaled)
sub = pd.DataFrame(columns=['seg_id','time_to_failure'])

sub['seg_id'] = y_test

sub['time_to_failure'] = y_pred_svr
filename = 'submission1.csv'

sub.to_csv(filename,index=False)

print('Saved file: ' + filename)