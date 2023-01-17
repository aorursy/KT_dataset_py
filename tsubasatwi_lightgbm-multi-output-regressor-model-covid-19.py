

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pandas_profiling import ProfileReport



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
train.head()
test.head()
submission
train.shape
test.shape
train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})

train_profile
test_profile = ProfileReport(test, title='Pandas Profiling Report', html={'style':{'full_width':True}})

test_profile
import numpy as np

import pandas as pd

import seaborn as sns



#for_modeling

import lightgbm as lgb



#for_plot

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import train_test_split





import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

from xgboost import XGBClassifier

import gc
train['month'] = train['Date'].str.extract(r'[0-9]+[-]([0-9]+)[-]')

train['day'] = train['Date'].str.extract(r'[0-9]+[-][0-9]+[-]([0-9]+)')

train = train.drop('Date', axis = 1)



test['month'] = test['Date'].str.extract(r'[0-9]+[-]([0-9]+)[-]')

test['day'] = test['Date'].str.extract(r'[0-9]+[-][0-9]+[-]([0-9]+)')

test = test.drop('Date', axis = 1)
def get_seed(x):

    return int(x[0:2])

train['month'] = train['month'].map(lambda x: get_seed(x))

train['day'] = train['day'].map(lambda x: get_seed(x))





def get_seed(x):

    return int(x[0:2])

test['month'] = test['month'].map(lambda x: get_seed(x))

test['day'] = test['day'].map(lambda x: get_seed(x))
x_train_drop = train.drop(['Province/State','Country/Region','ConfirmedCases','Fatalities'],axis=1)

y_train_drop = train.drop(['Province/State','Country/Region','Id','Province/State','Country/Region','Lat','Long','month','day'],axis=1)
test_drop = test.drop(['Province/State','Country/Region'],axis=1)
x_train_drop
y_train_drop
import lightgbm as lgb



from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



import numpy as np
import lightgbm as lgb



from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



from sklearn.multioutput import MultiOutputRegressor





import numpy as np
X_train, X_test, y_train, y_test = train_test_split(x_train_drop, y_train_drop)



# Train the model with the above parameters

# model = MultiOutputRegressor(lgb.LGBMRegressor()) #def

model = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=100, max_depth=-1,n_estimators=200,learning_rate=0.01))

model.fit(X_train, y_train)





# Predict divided learning data

y_pred = []

y_pred = model.predict(X_test)



# RMSE 

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print(rmse)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

r2
# testdata Precict

submission_pred = []

submission_pred = model.predict(test_drop)
submission_drop = submission.drop(['ConfirmedCases','Fatalities'],axis=1)
import pandas as pd

from pandas import Series, DataFrame
submission_drop_df = DataFrame(submission_drop)

submission_pred_df = DataFrame(submission_pred)
submission_pred_df.columns = ['ConfirmedCases', 'Fatalities']

submission_pred_df
submission_df = pd.merge(submission_drop_df, submission_pred_df,left_index=True,right_index=True)

submission_df
submission_df.to_csv("submission.csv", index=False)