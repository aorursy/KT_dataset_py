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

import numpy as np

from datetime import datetime

import sklearn.metrics

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split,cross_validate

from sklearn.metrics import make_scorer,precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score

import warnings

import plotly.offline as py

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
pd.set_option("display.max_rows",120)

pd.set_option("display.max_columns",120)
data=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv',index_col=0)

df_test=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv',index_col=0)

sample=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv')
data
df_test
null_feat = pd.DataFrame(len(data['Class']) - data.isnull().sum(), columns = ['Count'])



trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, marker=dict(color = 'lightgrey',

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  "Missing Values")

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
yes = data[(data['Class'] != 0)]

no = data[(data['Class'] == 0)]



#------------COUNT-----------------------

trace = go.Bar(x = (len(yes), len(no)), y = ['Yes', 'No'], orientation = 'h', opacity = 0.8, marker=dict(

        color=['gold', 'lightskyblue'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Count of Class variable')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)

#------------PERCENTAGE-------------------

trace = go.Pie(labels = ['No', 'Yes'], values = data['Class'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of Class variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
y=data['Class']

data=data.drop('Class',axis=1)

data=data.drop('Time',axis=1)

data=data.drop('Amount',axis=1)

X=data.iloc[:,0:30]
df_test=df_test.drop('Time',axis=1)

df_test=df_test.drop('Amount',axis=1)
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=42)

X_res, Y_res = sm.fit_sample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, Y_res, test_size = 0.3, random_state = 0)
dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
def xgb_opt(trial):

    X_train, X_test, y_train, y_test = train_test_split(X_res, Y_res, test_size = 0.3, random_state = 0)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {

        'silent': 1,

        'objective': 'binary:logistic',

        'eval_metric': 'auc',

        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),

        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),

        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)

    }



    if param['booster'] == 'gbtree' or param['booster'] == 'dart':

        param['max_depth'] = trial.suggest_int('max_depth', 1, 9)

        param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)

        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)

        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

    if param['booster'] == 'dart':

        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])

        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])

        param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)

        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)



    # Add a callback for pruning.

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-auc')

    bst = xgb.train(param, dtrain, evals=[(dtest, 'validation')], callbacks=[pruning_callback])

    preds = bst.predict(dtest)

    pred_labels = np.rint(preds)

    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)

    return accuracy
study=optuna.create_study()

study.optimize(xgb_opt,n_trials=100)
print(study.best_params)

print(study.best_value)

print(study.best_trial)
#xgb

b_param=study.best_params

bst = xgb.train(b_param, dtrain)

df_test_ar = df_test.as_matrix()

dm_d_test = xgb.DMatrix(df_test_ar)

preds = bst.predict(dm_d_test)
sample['Class']=preds

sample.to_csv('submit.csv',index=None)