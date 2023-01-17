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

import matplotlib.pyplot as plt



from sklearn import linear_model, ensemble

from sklearn.metrics import mean_squared_error, mean_absolute_error



import tensorflow as tf



from tqdm.notebook import tqdm



import os

from PIL import Image
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
train.head()
train.info()
test.head()
test.info()
train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
submission['Patient'] = (

    submission['Patient_Week']

    .apply(

        lambda x:x.split('_')[0]

    )

)



submission['Weeks'] = (

    submission['Patient_Week']

    .apply(

        lambda x: int(x.split('_')[-1])

    )

)



submission =  submission[['Patient','Weeks', 'Confidence','Patient_Week']]



submission = submission.merge(test.drop('Weeks', axis=1), on="Patient")
submission.head()
train['Dataset'] = 'train'

test['Dataset'] = 'test'

submission['Dataset'] = 'submission'
all_data = train.append([test, submission])



all_data = all_data.reset_index()

all_data = all_data.drop(columns=['index'])

all_data.head()



train_patients = train.Patient.unique()
fig, ax = plt.subplots(5, 1, figsize=(10, 20))



for i in range(5):

    patient_log = train[train['Patient'] == train_patients[i]]



    ax[i].set_title(train_patients[i])

    ax[i].plot(patient_log['Weeks'], patient_log['FVC'])
all_data['FirstWeek'] = all_data['Weeks']

all_data.loc[all_data.Dataset=='submission','FirstWeek'] = np.nan

all_data['FirstWeek'] = all_data.groupby('Patient')['FirstWeek'].transform('min')
first_fvc = (

    all_data

    .loc[all_data.Weeks == all_data.FirstWeek][['Patient','FVC']]

    .rename({'FVC': 'FirstFVC'}, axis=1)

    .groupby('Patient')

    .first()

    .reset_index()

)



all_data = all_data.merge(first_fvc, on='Patient', how='left')
all_data.head()
all_data['WeeksPassed'] = all_data['Weeks'] - all_data['FirstWeek']
all_data.head()
"""def calculate_height(row):

    if row['Sex'] == 'Male':

        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])

    else:

        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])



all_data['Height'] = all_data.apply(calculate_height, axis=1)

"""



def calculate_height(row):

    height = 0

    if row['Sex'] == 'Male' or 'Female':

        height = (((row['FirstFVC']/933.33) + 0.026*row['Age'] + 2.89)/0.0443)

        return int(height) 



all_data['Height'] = all_data.apply(calculate_height, axis=1)





def FEV1(row):

    FEV = 0

    if row['Sex'] == 'Male':

        FEV = (0.84 * row['FirstFVC'] - 0.23)

    else:

        FEV = (0.84 * row['FirstFVC'] - 0.36)

    return FEV

all_data['FEV'] = all_data.apply(FEV1, axis = 1)
all_data.head()
all_data = pd.concat([

    all_data,

    pd.get_dummies(all_data.Sex),

    pd.get_dummies(all_data.SmokingStatus)

], axis=1)



all_data = all_data.drop(columns=['Sex', 'SmokingStatus'])
all_data.head()
def scale_feature(series):

    return (series - series.min()) / (series.max() - series.min())



all_data['Weeks'] = scale_feature(all_data['Weeks'])

all_data['Percent'] = scale_feature(all_data['Percent'])

all_data['Age'] = scale_feature(all_data['Age'])

all_data['FirstWeek'] = scale_feature(all_data['FirstWeek'])

all_data['FirstFVC'] = scale_feature(all_data['FirstFVC'])

all_data['WeeksPassed'] = scale_feature(all_data['WeeksPassed'])

all_data['Height'] = scale_feature(all_data['Height'])

all_data['FEV'] = scale_feature(all_data['FEV'])
feature_columns = [

    'Percent',

    'Age',

    'FirstWeek',

    'FirstFVC',

    'WeeksPassed',

    'Female',

    'Male', 

    'Currently smokes',

    'Ex-smoker',

    'Never smoked',

    'FEV'

]
train = all_data.loc[all_data.Dataset == 'train']

test = all_data.loc[all_data.Dataset == 'test']

submission = all_data.loc[all_data.Dataset == 'submission']
train[feature_columns].head()
from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score



def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)

n_folds = 10



def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=1234).get_n_splits(train[feature_columns])

    rmse= np.sqrt(-cross_val_score(model, train[feature_columns], train['FVC'], scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.linear_model import Lasso

from sklearn.pipeline import make_pipeline 

from sklearn.linear_model import Lasso

lasso = Lasso()



lasso.fit(train[feature_columns], train['FVC'])
lasso_preds = lasso.predict(train[feature_columns])

sub = pd.DataFrame()

sub['lasso_FVC'] = lasso_preds

sub.head()
from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(train[feature_columns], train['FVC'])
!pip install ngboost
from ngboost import NGBRegressor

ngb = NGBRegressor()

ngb.fit(train[feature_columns], train['FVC'])
ngb_preds = ngb.predict(train[feature_columns])

sub = pd.DataFrame()

sub['ngb_FVC'] = ngb_preds

sub.head()
ridge_preds = lasso.predict(train[feature_columns])

sub = pd.DataFrame()

sub['ridge_FVC'] = ridge_preds

sub.head()
from sklearn.linear_model import BayesianRidge

bayesian_ridge = BayesianRidge()

bayesian_ridge.fit(train[feature_columns], train['FVC'])
bayesian_ridge_preds = bayesian_ridge.predict(train[feature_columns])

sub = pd.DataFrame()

sub['bay_ridge_FVC'] = bayesian_ridge_preds

sub.head()
from sklearn.linear_model import HuberRegressor

Huber = HuberRegressor()

Huber.fit(train[feature_columns], train['FVC'])
Huber_preds = Huber.predict(train[feature_columns])

sub = pd.DataFrame()

sub['huber_FVC'] = Huber_preds

sub.head()
from catboost import CatBoostRegressor

cat = CatBoostRegressor()

cat.fit(train[feature_columns], train['FVC'])
cat_preds = cat.predict(train[feature_columns])

sub = pd.DataFrame()

sub['cat_FVC'] = cat_preds

sub.head()
ridge_weight = 0.2

lasso_weight = 0.2

#cat_weight = 0.3

huber_weight = 0.40 

bayesian_ridge_weight = 0.20 
prediction1 = 0

sub = pd.DataFrame()

sub['ensembled_FVC'] = (ridge_preds*ridge_weight) + (lasso_preds*lasso_weight)  + (Huber_preds*huber_weight ) + (bayesian_ridge_preds*bayesian_ridge_weight)

predictions = sub['ensembled_FVC'].values
mse = mean_squared_error(

    train['FVC'],

    predictions,

    squared=False

)



mae = mean_absolute_error(

    train['FVC'],

    predictions

)



print('MSE Loss: {0:.2f}'.format(mse))

print('MAE Loss: {0:.2f}'.format(mae))
def competition_metric(trueFVC, predFVC, predSTD):

    clipSTD = np.clip(predSTD, 70 , 9e9)  

    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  

    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))

    



print(

    'Competition metric: ', 

    competition_metric(train['FVC'].values, predictions, 285) 

)
train['prediction'] = predictions
plt.scatter(predictions, train['FVC'])



plt.xlabel('predictions')

plt.ylabel('FVC (labels)')

plt.show()
delta = predictions - train['FVC']

plt.hist(delta, bins=20)

plt.show()
fig, ax = plt.subplots(5, 1, figsize=(10, 20))



for i in range(5):

    patient_log = train[train['Patient'] == train_patients[i]]



    ax[i].set_title(train_patients[i])

    ax[i].plot(patient_log['WeeksPassed'], patient_log['FVC'], label='truth')

    ax[i].plot(patient_log['WeeksPassed'], patient_log['prediction'], label='prediction')

    ax[i].legend()
submission[feature_columns].head()
lasso_preds1 = lasso.predict(submission[feature_columns])

submission1 = pd.DataFrame()

submission1['lasso_FVC'] = lasso_preds1

submission1.head()
ridge_preds1 = lasso.predict(submission[feature_columns])

submission2 = pd.DataFrame()

submission2['ridge_FVC'] = ridge_preds1

submission2.head()

bayesian_ridge_preds1 = bayesian_ridge.predict(submission[feature_columns])

submission3 = pd.DataFrame()

submission3['bay_ridge_FVC'] = bayesian_ridge_preds1

submission3.head()
Huber_preds1 = Huber.predict(submission[feature_columns])

submission4 = pd.DataFrame()

submission4['huber_FVC'] = Huber_preds1

submission4.head()
cat_preds1 = cat.predict(submission[feature_columns])

submission5 = pd.DataFrame()

submission5['cat_FVC'] = cat_preds1

submission5.head()
submission_ensemble = pd.DataFrame()

submission_ensemble['pred_FVC'] = (submission2['ridge_FVC'].values*ridge_weight) + (submission1['lasso_FVC']*lasso_weight)  + (submission4['huber_FVC'].values*huber_weight ) + (submission3['bay_ridge_FVC'].values*bayesian_ridge_weight)



final_prediction = submission_ensemble['pred_FVC'].values
submission_ensemble.shape 

submission['FVC'] = final_prediction
submission.head()
test_patients = list(submission.Patient.unique())

fig, ax = plt.subplots(5, 1, figsize=(10, 20))



for i in range(5):

    patient_log = submission[submission['Patient'] == test_patients[i]]



    ax[i].set_title(test_patients[i])

    ax[i].plot(patient_log['WeeksPassed'], patient_log['FVC'])
submission = submission[['Patient_Week', 'FVC']]



submission['Confidence'] = 275
print(len(submission['FVC'].unique()))
submission.to_csv('submission.csv', index=False)
submission.head()
def competition_metric(trueFVC, predFVC, predSTD):

    clipSTD = np.clip(predSTD, 70 , 9e9)  

    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  

    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))

    



print(

    'Competition metric: ', 

    competition_metric(submission['FVC'].values, final_prediction, 275) 

)