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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import lightgbm as lgb

import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Other Libraries

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")
# Importing the dataset

df = pd.read_csv("../input/creditcard.csv")

df.head()

df.describe()

print(round(df['Class'].value_counts()[0]/len(df) * 100,2))

print(round(df['Class'].value_counts()[1]/len(df) * 100,2))
#Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



#RobustScaler is less prone to outliers.

std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)



scaled_amount = df['scaled_amount']

scaled_time = df['scaled_time']



df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount) 

df.insert(1, 'scaled_time', scaled_time)



X = df.drop('Class', axis=1)

y = df.iloc[:, 30].values

df2=df.copy() 

df2.drop(df2.columns[14], axis = 1, inplace = True)

df2.drop(df2.columns[15], axis = 1, inplace = True)
X = df2.drop('Class', axis=1)

y = df2.iloc[:, 28].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred_drop2 = classifier.predict(X_test)

print(roc_auc_score(y_test, y_pred_drop2))

print(accuracy_score(y_test, y_pred_drop2))

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_l = confusion_matrix(y_test, y_pred_drop2)

print(cm_l)
# test classification dataset

from sklearn.datasets import make_classification

# test regression dataset

from sklearn.datasets import make_regression

from xgboost import XGBRegressor

# define dataset

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)

# define the model

model = XGBRegressor()

# fit the model

model.fit(X, y)

# get importance

importance = model.feature_importances_

# summarize feature importance

for i,v in enumerate(importance):

	print('Feature: %0d, Score: %.5f' % (i,v))
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



# amount of fraud classes 492 rows.

fraud_df = df2.loc[df['Class'] == 1]

non_fraud_df = df2.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df
X_t=new_df.drop('Class', axis=1)

y_t =new_df.iloc[:, 28].values
fraud_df_tt = df2.loc[df2['Class'] == 1][369:492]

non_fraud_df_tt = df2.loc[df2['Class'] == 0][369:]

undersampled_df_test = pd.concat([fraud_df_tt, non_fraud_df_tt])

undersampled_df_test = sklearn.utils.shuffle(undersampled_df_test)

X_tt=undersampled_df_test.drop('Class', axis=1)

y_tt = undersampled_df_test.iloc[:, 28].values
#XGB

gg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=2,

              min_child_weight=1, missing=None, n_estimators=70, n_jobs=-1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

gg.fit(X_t, y_t)

y_pred_dropxg = gg.predict_proba(X_tt)[:, 1]
len(y_pred_dropxg)

y_pred_dropxgb=np.zeros(284069, dtype = int) 

for i in range(len(y_pred_dropxg)):

        if y_pred_dropxg[i] > 0.75:

            y_pred_dropxgb[i]=1

print(roc_auc_score(y_tt, y_pred_dropxgb))



accuracy_score(y_tt, y_pred_dropxgb)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_xgb = confusion_matrix(y_tt, y_pred_dropxgb)

cm_xgb
fraud_df_tt = df2.loc[df2['Class'] == 1][369:492] 

non_fraud_df_tt = df2.loc[df2['Class'] == 0][369:] #ratio in test case same 

undersampled_df_test = pd.concat([fraud_df_tt, non_fraud_df_tt])

undersampled_df_test = sklearn.utils.shuffle(undersampled_df_test)

X_tt=undersampled_df_test.drop('Class', axis=1)

y_tt = undersampled_df_test.iloc[:, 28].values
df2 = sklearn.utils.shuffle(df2)

fraud_df = df2.loc[df2['Class'] == 1][:369] 

non_fraud_df = df2.loc[df2['Class'] == 0][:150000] 

undersampled_df_train = pd.concat([fraud_df, non_fraud_df])

undersampled_df_train = sklearn.utils.shuffle(undersampled_df_train)

X_t=undersampled_df_train.drop('Class', axis=1)

y_t = undersampled_df_train.iloc[:, 28].values
# Oversample and plot imbalanced dataset with SMOTE

from collections import Counter

from sklearn.datasets import make_classification

from imblearn.over_sampling import SMOTE

from matplotlib import pyplot

from numpy import where

# define dataset

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,

	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

oversample = SMOTE()

X, y = oversample.fit_resample(X, y)
#Oversample

from imblearn.over_sampling import SMOTE

def plot_2d_space(X_t, y_t, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()

    

smote = SMOTE()

X_sm, y_sm = smote.fit_sample(X_t, y_t)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
#XGB

gg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=2,

              min_child_weight=1, missing=None, n_estimators=70, n_jobs=-1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

gg.fit(X_sm, y_sm)

y_pred_overxgb = gg.predict_proba(X_tt)[:, 1]
outcome = pd.DataFrame(y_pred_overxgb)

outcome.to_csv('y_prob_overxgb.csv', index=False)

len(y_pred_overxgb)
y_pred_dropxgov=np.zeros(284069, dtype = int) 

for i in range(len(y_pred_overxgb)):

        if y_pred_overxgb[i] > 0.95:

            y_pred_dropxgov[i]=1
print(roc_auc_score(y_tt, y_pred_dropxgov))

accuracy_score(y_tt, y_pred_dropxgov)

cm_xg = confusion_matrix(y_tt, y_pred_dropxgov)

print(cm_xg)
#SVM

# Fitting SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_sm, y_sm)

y_pred_dropsv = classifier.predict(X_tt)
print(roc_auc_score(y_tt, y_pred_dropsv))

print(accuracy_score(y_tt, y_pred_dropsv))

cm_sv = confusion_matrix(y_tt, y_pred_dropsv)

print(cm_sv)
outcome = pd.DataFrame(y_pred_dropsv)

outcome.to_csv('y_prob_oversv.csv', index=False)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_sm, y_sm)



# Predicting the Test set results

y_predlr = classifier.predict(X_tt)
print(roc_auc_score(y_tt, y_predlr))

print(accuracy_score(y_tt, y_predlr))

cm_l = confusion_matrix(y_tt, y_predlr)

print(cm_l)
outcome = pd.DataFrame(y_predlr)

outcome.to_csv('y_pred_overlr.csv', index=False)
#Light GBM on df3

lgb_train = lgb.Dataset(X_sm, y_sm)

lgb_eval = lgb.Dataset(X_tt, y_tt, reference=lgb_train)



params = {

        'objective': 'binary',

        'feature_fraction': 1,

        'bagging_fraction': 1,

        'verbose': -1

    }



gbm = lgb.train(params,lgb_train,num_boost_round=20)

y_pred_gbm = gbm.predict(X_tt)
y_pred_overgbm=np.zeros(len(y_pred_gbm), dtype = int) 

for i in range(len(y_pred_gbm)):

        if y_pred_gbm[i] > 0.95:

            y_pred_overgbm[i]=1
print(roc_auc_score(y_tt, y_pred_overgbm))

print(accuracy_score(y_tt, y_pred_overgbm))

cm_gbm = confusion_matrix(y_tt, y_pred_overgbm)

print(cm_gbm)
outcome = pd.DataFrame(y_pred_overgbm)

outcome.to_csv('y_pred_overgbm.csv', index=False)