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
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import xgboost as xgb 
import os

print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']

train_y = df_train.loc[:, 'O/P']
#xgb used as it is best

#n estimatior is used to make it learn more time to reduse error and max_depth for using nodes learning rate is to take time to learn that

#rf=RandomForestRegressor(n_estimators=200,random_state=40,max_features='auto',min_samples_leaf=4,max_depth=70,bootstrap='True')#

XGB=xgb.XGBRegressor(n_estimators=2000,max_depth=6,learning_rate=0.1,gamma=0,random_state=2,verbosity=2)
XGB.fit(train_X,train_y)

score=XGB.score(train_X,train_y)

print(score)
predc =XGB.predict(train_X)

print(predc)
df_test = df_test.loc[:, 'F3':'F17']

pred =XGB.predict(df_test)
print(pred)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('output.csv', index=False)
