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
#train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)

#test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)

train_df = pd.read_csv('/kaggle/input/dataset2/train2.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/dataset2/test2.csv', index_col=0)
train_df
test_df
train_df.dtypes
test_df.dtypes


train_df = train_df.replace('?', np.NaN)

test_df = test_df.replace('?', np.NaN)



train_df=train_df.fillna(train_df.mode().T[0])

#train_df=train_df.fillna(train_df.mean)

#train_df=train_df.dropna(how='all', axis=1)

#train_df=train_df.fillna(0)





test_df=test_df.fillna(test_df.mode().T[0])

#test_df=test_df.fillna(test_df.mean)

#test_df=test_df.dropna(how='all', axis=1)

#test_df=test_df.fillna(0)





train_df=train_df.astype({'normalized-losses': 'float64'})

train_df=train_df.astype({'bore': 'float64'})

train_df=train_df.astype({'stroke': 'float64'})

train_df=train_df.astype({'horsepower': 'float64'})

train_df=train_df.astype({'peak-rpm': 'float64'})

train_df=train_df.astype({'price': 'float64'})



test_df=test_df.astype({'normalized-losses': 'float64'})

test_df=test_df.astype({'bore': 'float64'})

test_df=test_df.astype({'stroke': 'float64'})

test_df=test_df.astype({'horsepower': 'float64'})

test_df=test_df.astype({'peak-rpm': 'float64'})

test_df=test_df.astype({'price': 'float64'})
numeric_columns = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'compression-ratio', 'city-mpg', 'highway-mpg']



#X_train = train_df[numeric_columns].to_numpy()

X_train = train_df[test_df.columns.values].to_numpy()



y_train = train_df['symboling'].to_numpy()



#X_test = test_df[numeric_columns].to_numpy()

X_test = test_df[test_df.columns.values].to_numpy()

X_train

from sklearn.model_selection import train_test_split



myX_train, myX_test, myy_train, myy_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score



'''

#mydataset1

model = MLPClassifier(learning_rate='invscaling',

                      validation_fraction=0.8,

                      activation='tanh',

                      batch_size=50,

                      #learning_rate_init=0.0000005,

                      learning_rate_init=0.000005,

                      max_iter=100000,

                      tol=0.0000001,

                      verbose='true',

                      #hidden_layer_sizes=(256,256,256,256,256,256,256,256,256,256,256,256,256))

                      hidden_layer_sizes=(16,16))

model.fit(X_train, y_train)



#mydataset2

model = MLPClassifier(learning_rate='invscaling',

                      validation_fraction=0.3,

                      activation='tanh',

                      batch_size=50,

                      learning_rate_init=0.000001,

                      max_iter=100000,

                      tol=0.00000001,

                      verbose='true',

                      hidden_layer_sizes=(80,80,80,80,80,80,80,80,80,80,80,80))

model.fit(X_train,y_train)

'''



from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

ypred=model.predict(X_train)

accuracy_score(ypred,y_train)
ypred=model.predict(myX_test)

accuracy_score(ypred,myy_test)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')