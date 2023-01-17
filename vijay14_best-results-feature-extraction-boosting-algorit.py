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
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns
# reading the data

data = pd.read_csv(r"../input/advertising/advertising.csv")

data.head()
data.shape
data.info()
data.describe()
# Renaming columns

data = data.rename(columns={'Daily Time Spent on Site': 'time_spent', 

                     'Area Income': 'area_income', 

                     'Daily Internet Usage': 'internet_usage',

                    'Ad Topic Line': 'ad_topic_line',

                     'Clicked on Ad': 'click_ad'

                    })
data.loc[lambda df: df.click_ad == 1].describe()
data.loc[lambda df: df.click_ad == 0].describe()
sns.pairplot(data, hue= 'click_ad')
# age group

age_map = {range(10, 20): 0,

          range(20, 30): 1,

          range(30,40): 2,

          range(40,50): 3,

          range(50, max(data.Age)): 4}

data['age_group'] = data.Age.apply(lambda x: next((v for k, v in age_map.items() if x in k),255))



# time based

data.Timestamp = pd.to_datetime(data.Timestamp)

data['month'] = data.Timestamp.dt.month

data['week'] = data.Timestamp.dt.week

data['wday'] = data.Timestamp.dt.weekday

data['dhour'] = data.Timestamp.dt.hour



# ratio of time spent on the website versus average internet usage per day

data['ratio_usage'] = data.time_spent / data.internet_usage
# defining features and data preprocessing steps

features = {"categorical": ['City', 'Country', 'Male', 'age_group', 'wday', 'month'],

           "numerical": ['area_income',  'internet_usage', 'time_spent'],

            "binary": ['Age', 'ratio_usage']

                      }

categories = []

for f in features["categorical"]:

    categories.append(np.unique(data[f]))

    

processing_type = {"categorical": OrdinalEncoder(categories=categories),

                  "numerical": StandardScaler(),

                   "binary": "passthrough"

          }

processor = ColumnTransformer([(t, processing_type[t], n) for t, n in features.items()])
model_params = {

        "max_depth": 6,

        "n_estimators": 400,

        "min_child_weight": 1,

        "subsample": .8,

        "objective": "reg:logistic",

        "tree_method": "exact",

        "n_jobs": 4,

        "learning_rate": 0.1,

        "colsample_bytree ": 0.8,

        "scale_pos_weight": 1,

        "gamma": 0 # 0, greater the value - higher the conservative the model is

 }

predictor = XGBClassifier()

model = Pipeline(steps= [('data_processor', processor), ('pca', PCA()), ('predictor', predictor)]) #('pcs', PCA())
# split train and test set

xtrain, xtest, ytrain, ytest = train_test_split(data, data.click_ad, test_size=0.2, random_state = 25)
# Model training 

model.fit(xtrain, ytrain)
# Training result

train_pred = model.predict(xtrain)

print("Training score(accuracy): ", accuracy_score(ytrain, train_pred))
# testing results

ypred = model.predict(xtest)

print("Test score(accuracy): ", accuracy_score(ytest, ypred))
print("classification report: ")

print(classification_report(ytest, ypred))
print("confusion matrix: ")

print(confusion_matrix(y_true=ytest, y_pred=ypred, labels=[0, 1]))