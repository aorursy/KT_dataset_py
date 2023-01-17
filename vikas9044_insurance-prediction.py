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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from matplotlib import pyplot

 
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report

from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_auc_score,accuracy_score
df1=pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
df1.head(n=10)
#we ll try look at the response variable if its balanced or impbalanced classification

sns.countplot(df1['Response'])

df1.dtypes
df1.isnull().sum()



numeric_cols=[]

categorical_cols=[]

def colmns_dtypes(dataframe):

    for i in dataframe.columns:

        if dataframe[i].dtypes!='object':

            numeric_cols.append(i)

        else:

            categorical_cols.append(i)

colmns_dtypes(df1)

            
numeric_cols
categorical_cols
fig, ax =plt.subplots(1,2,figsize=(15,5))

sns.countplot(x='Gender',hue='Previously_Insured',data=df1,ax=ax[0])

sns.countplot(x='Gender',hue='Vehicle_Damage',data=df1,ax=ax[1])

fig.show()
sns.stripplot(y='Annual_Premium',x='Response',data=df1)
fig, ax =plt.subplots(1,2,figsize=(15,5))

sns.countplot(data=df1,x='Gender',hue='Vehicle_Age',ax=ax[0])

sns.countplot(data=df1,x='Previously_Insured',hue='Vehicle_Damage',ax=ax[1])

fig.show()
plt.figure(figsize=(20,9))

sns.FacetGrid(df1, hue = 'Response',

             height = 6,xlim = (0,150)).map(sns.kdeplot, 'Age', shade = True,bw=2).add_legend()
df1.describe()
le = LabelEncoder()

df1['Gender'] = le.fit_transform(df1['Gender'])

df1['Driving_License'] = le.fit_transform(df1['Driving_License'])

df1['Previously_Insured'] = le.fit_transform(df1['Previously_Insured'])

df1['Vehicle_Damage'] = le.fit_transform(df1['Vehicle_Damage'])

df1['Driving_License'] = le.fit_transform(df1['Driving_License'])

df1['Vehicle_Age'] = le.fit_transform(df1['Vehicle_Age'])
X2 = df1.drop(["Response"], axis=1)

y2 = df1["Response"]

def select_features(X_train, y_train, X_test):

    fs = SelectKBest(score_func=f_classif, k='all')

    fs.fit(X_train, y_train)

    X_train_fs = fs.transform(X_train)

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33, random_state=1)

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)



for i in range(len(fs.scores_)):

    print('Feature %d: %f' % (i, fs.scores_[i]))



pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)

pyplot.show()
df1.columns
def evaluation_stats(model,X_train, X_test, y_train, y_test,algo,is_feature=False):

    print('Train Accuracy')

    y_pred_train = model.predict(X_train)                           

    print(accuracy_score(y_train, y_pred_train))

    print('Validation Accuracy')

    y_pred_test = model.predict(X_test)                           

    print(accuracy_score(y_test, y_pred_test))

    print("\n")

    print("Train AUC Score")

    print(roc_auc_score(y_train, y_pred_train))

    print("Test AUC Score")

    print(roc_auc_score(y_test, y_pred_test))

    

def training(model,X_train, y_train):

    return model.fit(X_train, y_train)
sm = SMOTE(random_state=101)

X_res, y_res = sm.fit_resample(X_train, y_train)
rf_model = training(RandomForestClassifier(),X_res,y_res)

evaluation_stats(rf_model,X_res, X_test, y_res, y_test,'RANDOM FOREST')
rf_model = training(RandomForestClassifier(criterion='entropy',n_estimators=200,max_depth=3),X_res, y_res)

evaluation_stats(rf_model,X_res, X_test, y_res, y_test,'RANDOM FOREST')
from sklearn.linear_model import LogisticRegression
lr_model = training(LogisticRegression(),X_res,y_res)

evaluation_stats(lr_model,X_res, X_test, y_res, y_test,'logistic regression')

grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}

grid_clf_acc = GridSearchCV(lr_model, param_grid = grid_values,scoring = 'recall')







grid_clf_acc.fit(X_train, y_train)
y_pred_acc = grid_clf_acc.predict(X_test)



print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))

print('Test AUC Score : ' + str(roc_auc_score(y_test,y_pred_acc)))
xbg_model = training(XGBClassifier(n_estimators=1000,max_depth=10),X_res, y_res)

evaluation_stats(xbg_model,X_res, X_test, y_res, y_test,'XGB',is_feature=False)