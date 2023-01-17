import pandas as pd

import numpy as np

import os

import joblib

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
input_path = '../input/1056lab-fraud-detection-in-credit-card/'

os.listdir(input_path)
target_col = 'Class'

df_train = pd.read_csv(os.path.join(input_path, 'train.csv'), index_col=0)

df_test = pd.read_csv(os.path.join(input_path, 'test.csv'), index_col=0)
from sklearn.base import BaseEstimator, TransformerMixin

from itertools import combinations as comb

class CreateCalc(BaseEstimator, TransformerMixin):

    

    def __init__(self, return_df=True):

        self.return_df_ = return_df

    

    def _calc(self, data, col1, col2, operator, reverse=False):

        if reverse:

            col1, col2 = col2, col1

            

        if operator == "+":

            return data[col1] + data[col2]

        elif operator == "-":

            return data[col1] - data[col2]

        elif operator == "*":

            return data[col1] * data[col2]

        elif operator == "/":

            return data[col1] / data[col2]

        elif operator == "//":

            return data[col1] // data[col2]

        else:

            print("その演算子は対応していません。")

            return None

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, operators=["+", "-", "*", "/"]):

        cols = X.columns

        X_ = X.copy()

        

        for col1, col2 in list(comb(cols, 2)):

            for operator in operators:

                X_["{} {} {}".format(col1, operator, col2)] = self._calc(X_, col1, col2, operator)

        

        if self.return_df:

            return X_

        else:

            return X_.values
df_train.head()
df_test.head()
sns.countplot(df_train['Class'])

df_train['Class'].value_counts()
df_train.info()
# for col in df_test.columns:

#     plt.figure(figsize=(20, 5))

#     sns.distplot(df_train[col], label='Train')

#     sns.distplot(df_test[col], label='Test')

#     plt.legend()

#     plt.show()
# plt.hist(df_train['Time'], label='Train')

# plt.hist(df_test['Time'], label='Test')

sns.distplot(df_train["Time"], label="Train")

sns.distplot(df_test["Time"], label="Test")

plt.legend()

plt.show()
df_train.corr()['Class'].sort_values()
df_train.isnull().sum()
df_test.isnull().sum()
# for col in df_test.columns:

#     plt.figure(figsize=(20, 5))

#     sns.distplot(df_train[df_train[target_col] == 0][col], label='Class 0')

#     sns.distplot(df_train[df_train[target_col] == 1][col], label='Class 1')

#     plt.legend()

#     plt.show()
from imblearn.over_sampling import SMOTE



smote = SMOTE()

tmp_X = df_train.drop([target_col], axis=1)

tmp_y = df_train[target_col]



tmp_X, tmp_y = smote.fit_sample(tmp_X, tmp_y)



from lightgbm import LGBMClassifier



clf = LGBMClassifier()

clf.fit(tmp_X, tmp_y)



dic_imp = dict([x for x in zip(df_test.columns, clf.feature_importances_)])

sorted_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=True))

df_feat_importance = pd.DataFrame([sorted_imp.keys(), sorted_imp.values()], index=["Feature", "Impotance"]).T



plt.figure(figsize=(12, 24))

sns.barplot(x="Impotance", y="Feature", data=df_feat_importance)
target_col = 'Class'

df_train = pd.read_csv(os.path.join(input_path, 'train.csv'), index_col=0)

df_test = pd.read_csv(os.path.join(input_path, 'test.csv'), index_col=0)
import pandas as pd

import numpy as np

from tqdm.notebook import tqdm



def mem_usage(df):

    print(df)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    col_type = df.dtypes

    if col_type in numerics:

        c_min = df.min()

        c_max = df.max()

        if str(col_type)[:3] == 'int':

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df_ = df.astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df_ = df.astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df_ = df.astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df_ = df.astype(np.int64)

        else:

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df_ = df.astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df_ = df.astype(np.float32)

            else:

                df_ = df.astype(np.float64)

    return df_
import math



df_train["Amount_dec"] = df_train["Amount"].apply(lambda x: math.modf(x)[0])

df_train["Amount_int"] = df_train["Amount"].apply(lambda x: math.modf(x)[1])



df_test["Amount_dec"] = df_test["Amount"].apply(lambda x: math.modf(x)[0])

df_test["Amount_int"] = df_test["Amount"].apply(lambda x: math.modf(x)[1])



df_train.drop(["Amount"], axis=1, inplace=True)

df_test.drop(["Amount"], axis=1, inplace=True)
float_cols = df_test.drop(['Time'], axis=1).columns

df_train[float_cols] = df_train[float_cols].astype(np.float32)

df_test[float_cols] = df_test[float_cols].astype(np.float32)

df_train['Time'] = df_train['Time'].astype(np.int32)

df_test['Time'] = df_test['Time'].astype(np.int32)
from itertools import combinations as comb



cols = [col for col in df_test.columns if col.startswith('V')]

for col_1, col_2 in tqdm(list(comb(cols, 2))):

    # Bacause memory usage>16GB, under 4 lines are commented out.

#     df_train['{} + {}'.format(col_1, col_2)] = df_train[col_1] + df_train[col_2]

#     df_test['{} + {}'.format(col_1, col_2)] = df_test[col_1] + df_test[col_2]

    

#     df_train['{} - {}'.format(col_1, col_2)] = df_train[col_1] - df_train[col_2]

#     df_test['{} - {}'.format(col_1, col_2)] = df_test[col_1] - df_test[col_2]

    

    df_train['{} * {}'.format(col_1, col_2)] = df_train[col_1] * df_train[col_2]

    df_test['{} * {}'.format(col_1, col_2)] = df_test[col_1] * df_test[col_2]

    

    df_train['{} / {}'.format(col_1, col_2)] = df_train[col_1] / df_train[col_2]

    df_test['{} / {}'.format(col_1, col_2)] = df_test[col_1] / df_test[col_2]

    

    df_train['{} / {}'.format(col_2, col_1)] = df_train[col_2] / df_train[col_1]

    df_test['{} / {}'.format(col_2, col_1)] = df_test[col_2] / df_test[col_1]
# float_cols = df_test.drop(['Time'], axis=1).columns

# df_train[float_cols] = df_train[float_cols].astype(np.float16)

# df_test[float_cols] = df_test[float_cols].astype(np.float16)

df_train.info()
df_train.shape
import gc

gc.collect()
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE



cv = 5



X = df_train.drop([target_col], axis=1).values

y = df_train[target_col].values



estimators = {

    'LightGBM': LGBMClassifier(),

#     'XGBoost': XGBClassifier()

}



k_fold = KFold(n_splits=cv, shuffle=True, random_state=42)

for clf_name, clf in estimators.items():

#     print(clf_name, ':', end=' ')

    cv_score = np.zeros(len(X))

    for fold, ids in enumerate(k_fold.split(X, y)):

#         print('{} Fold'.format(fold+1))

        

        X_train, y_train = X[ids[0]], y[ids[0]]

        X_valid, y_valid = X[ids[1]], y[ids[1]]

        

        smote = SMOTE()

        X_train, y_train = smote.fit_sample(X_train, y_train)

        

        clf.fit(X_train, y_train)

        cv_score[ids[1]] += clf.predict_proba(X_valid)[:, 1]

    print(clf_name, ":", roc_auc_score(y, cv_score))
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE



smote = SMOTE()

tmp_X = df_train.drop([target_col], axis=1)

tmp_y = df_train[target_col]



clf = LGBMClassifier()

clf.fit(tmp_X, tmp_y)

del tmp_X, tmp_y; gc.collect()



dic_imp = dict([x for x in zip(df_train.drop([target_col], axis=1).columns, clf.feature_importances_)])

sorted_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=True))

df_feat_importance = pd.DataFrame([sorted_imp.keys(), sorted_imp.values()], index=["Feature", "Impotance"]).T



plt.figure(figsize=(12, 24))

sns.barplot(x="Impotance", y="Feature", data=df_feat_importance[: 40])
top_feat = df_feat_importance.iloc[:40]["Feature"].values
from lightgbm import LGBMClassifier



X = df_train.drop([target_col], axis=1).values

y = df_train[target_col].values

X_test = df_test.values



X_top_feat = df_train[top_feat].values

X_test_top = df_test[top_feat].values



del df_train, df_test; gc.collect()
from imblearn.over_sampling import SMOTE



smote = SMOTE()



X_sampled, y_sampled = smote.fit_sample(X, y)

X_samp_top , y_samp_top = smote.fit_sample(X_top_feat, y)



gc.collect()
clf = LGBMClassifier()

clf.fit(X_sampled, y_sampled)

predict = clf.predict_proba(X_test)[:, 1]
submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv'))

submit[target_col] = predict

submit.to_csv('submit.csv', index=False)