import pandas as pd

import numpy as np

from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split,cross_validate

from sklearn.metrics import make_scorer,precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score,mean_squared_error

import warnings

import plotly.offline as py

import math

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split

import optuna

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,mean_squared_log_error

warnings.filterwarnings('ignore')



from sklearn.svm import SVC

import lightgbm as lgb

import xgboost as xgb
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
pd.set_option("display.max_rows",120)

pd.set_option("display.max_columns",120)
data=pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv',index_col=0)

df_test=pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv',index_col=0)

sample=pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv')
data
df_test
null_feat = pd.DataFrame(len(data['G3']) - data.isnull().sum(), columns = ['Count'])



trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, marker=dict(color = 'lightgrey',

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  "Missing Values")

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
dummies_data=pd.get_dummies(data[['school','class','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']])

dummies_test=pd.get_dummies(df_test[['school','class','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']])
data = data.merge(dummies_data, left_index=True, right_index=True)

data = data.drop(['school','class','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian'],axis=1 )

df_test= df_test.merge(dummies_test, left_index=True, right_index=True)

df_test = df_test.drop(['school','class','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian'],axis=1 )
def fill_missing_columns(df_a, df_b):

    columns_for_b = set(df_a.columns) - set(df_b.columns)

    for column in columns_for_b:

        df_b[column] = 0

    columns_for_a = set(df_b.columns) - set(df_a.columns)

    for column in columns_for_a:

        df_a[column] = 0
y=data['G3']

data=data.drop('G3',axis=1)
fill_missing_columns(data, df_test)
X=data.iloc[:,0:58]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
dm_train = xgb.DMatrix(X_train, label=y_train)

dm_test = xgb.DMatrix(X_test, label=y_test)
xgboost_tuna = xgb.XGBRegressor(random_state=42)
def xgb_opt(trial):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    dm_train = xgb.DMatrix(X_train, label=y_train)

    dm_test = xgb.DMatrix(X_test, label=y_test)

    param={

        'objective': 'reg:linear',

        'silent': 1,

        'random_state':42,

        'n_estimators' :trial.suggest_int('n_estimators', 0, 1000),

        'max_depth' : trial.suggest_int('max_depth', 1, 25),

        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 20),

        'subsample' : trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1),

        'gamma':trial.suggest_uniform('gamma', 0.0, 10.0),

        'eta':trial.suggest_loguniform('eta',1e-8,1.0),

    }

    bst = xgb.train(param, dm_train)

    preds = bst.predict(dm_test)

    pred_labels = np.rint(preds)

    rmse=np.sqrt(mean_squared_error( y_test, pred_labels))

    return rmse
study=optuna.create_study()

study.optimize(xgb_opt,n_trials=200)
print(study.best_params)

print(study.best_value)

print(study.best_trial)
#xgb

b_param=study.best_params

bst = xgb.train(b_param, dm_train)

dm_d_test = xgb.DMatrix(df_test)

preds = bst.predict(dm_d_test)
sample['G3']=preds

sample.to_csv('submit.csv',index=None)