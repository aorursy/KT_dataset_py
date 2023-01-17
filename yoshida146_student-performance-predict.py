!pip install -U imbalanced-learn
import pandas as pd

import numpy as np

import os

import joblib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from tqdm.notebook import tqdm

import tensorflow as tf
import sys, matplotlib, sklearn, lightgbm, xgboost, imblearn

import tqdm as tqdm_v



print("Python version :", sys.version)

dic_ver = {

    "Pandas": pd.__version__,

    "NumPy": np.__version__,

    "Joblib": joblib.__version__,

    "Matplotlib": matplotlib.__version__,

    "Seaborn": sns.__version__,

    "tqdm": tqdm_v.__version__,

    "Scikit-Learn": sklearn.__version__,

    "Imbalanced Learn": imblearn.__version__, 

    "LightGBM": lightgbm.__version__,

    'XGBoost': xgboost.__version__,

    'TensorFlow': tf.__version__,

}

del sys, matplotlib, sklearn, lightgbm, tqdm_v, xgboost, imblearn

pd.DataFrame(dic_ver.values(), index=dic_ver.keys(), columns=["Version"])
from sklearn.metrics import mean_squared_error, make_scorer

def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse)
input_path = os.path.join('..', 'input', '1056lab-student-performance-prediction')



target_col = 'G3'

df_train = pd.read_csv(os.path.join(input_path, 'train.csv'), index_col='id')

df_test = pd.read_csv(os.path.join(input_path, 'test.csv'), index_col='id')
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
sns.distplot(df_train[target_col])

# plt.savefig('target.pdf')

plt.show()
sns.countplot(df_train[target_col])

plt.show()
obj_cols = [col for t, col in zip(df_test.dtypes, df_test.columns) if t == 'object']

obj_cols
flag = True

for col in obj_cols:

    if len(set(df_train[col].unique()) - set(df_test[col].unique())) > 0:

        print(col, 'の値の数が一致しません')

        flag = False



if flag:

    print('値の数が一致しない列は存在しませんでした')
for col in obj_cols:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    tmp_train_vc = df_train[col].value_counts()

    ax1.pie(tmp_train_vc, labels=tmp_train_vc.index, autopct='%1.1f%%')



    tmp_test_vc = df_test[col].value_counts()

    ax2.pie(tmp_test_vc, labels=tmp_test_vc.index, autopct='%1.1f%%')

    

    ax1.title.set_text('{} Percentage.'.format(col))

    ax2.title.set_text('{} Percentage.'.format(col))
for col in obj_cols:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    

    order_col = df_train[col].value_counts().index

    sns.countplot(df_train[col], order=order_col, ax=ax1)

    sns.countplot(df_test[col], order=order_col, ax=ax2)
num_cols = [col for t, col in zip(df_test.dtypes, df_test.columns) if t == 'float' or t == 'int'] 

num_cols
for col in num_cols:

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    fig = plt.figure(figsize=(10, 5))

    

    sns.distplot(df_train[col], label='Train')

    sns.distplot(df_test[col], label='Test')

    plt.legend()

    plt.show()
for col in num_cols:

#     plt.figure()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    ax1.hist(df_train[col], label="Train", density=True)

    ax2.hist(df_test[col], label="Test", density=True)

    plt.title(col)

    plt.show()
plt.hist(df_train['failures'], density=True, label='Train', alpha=0.3)

plt.hist(df_test['failures'], density=True, label='Test', alpha=0.3)
bool_cols = [col for t, col in zip(df_test.dtypes, df_test.columns) if t == 'bool'] 

bool_cols
for col in bool_cols:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    

    sns.countplot(df_train[col], label='Train', ax=ax1)

    sns.countplot(df_test[col], label='Test', ax=ax2)

    plt.show()
for col in bool_cols:

    if len(df_train[df_train[target_col] > 14][col].value_counts()) <= 1:

        print(col)
df_train.corr()[target_col].sort_values()
cat_cols = obj_cols + bool_cols
df_train_, df_test_ = df_train.copy(), df_test.copy()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



obj_cols = [col for t, col in zip(df_test.dtypes, df_test.columns) if t == 'object']

for col in obj_cols:

    df_train_[col] = le.fit_transform(df_train_[col])

    df_test_[col] = le.transform(df_test_[col])
df_train_['All_Sup'] = df_train_['famsup'] & df_train_['schoolsup']

df_test_['All_Sup'] = df_test_['famsup'] & df_test_['schoolsup']
df_train_['PairEdu'] = df_train_[['Fedu', 'Medu']].mean(axis=1)

df_test_['PairEdu'] = df_test_[['Fedu', 'Medu']].mean(axis=1)
df_train_['more_high'] = df_train['higher'] & (df_train['schoolsup'] | df_train['paid'])

df_test_['more_high'] = df_test['higher'] & (df_test['schoolsup'] | df_test['paid'])
df_train_['All_alc'] = df_train_['Walc'] + df_train_['Dalc']

df_test_['All_alc'] = df_test_['Walc'] + df_test_['Dalc']



df_train_['Dalc_per_week'] = df_train_['Dalc'] / df_train_['All_alc']

df_test_['Dalc_per_week'] = df_test_['Dalc'] / df_test_['All_alc']



# スコアが下がったので実行しない

# df_train_['Walc_per_week'] = df_train_['Walc'] / df_train_['All_alc']

# df_test_['Walc_per_week'] = df_test_['Walc'] / df_test_['All_alc']



df_train_.drop(['Dalc'], axis=1, inplace=True)

df_test_.drop(['Dalc'], axis=1, inplace=True)
df_train_['studytime_ratio'] = df_train_['studytime'] / (df_train_[['studytime', 'traveltime', 'freetime']].sum(axis=1))

df_test_['studytime_ratio'] = df_test_['studytime'] / (df_test_[['studytime', 'traveltime', 'freetime']].sum(axis=1))



df_train_.drop(["studytime"], axis=1, inplace=True)

df_test_.drop(["studytime"], axis=1, inplace=True)
target_0_indx = df_train[df_train[target_col] == 0].index
from sklearn.model_selection import KFold

from tqdm.notebook import tqdm

import re

from functools import partial



def cross_validation(reg, X, y, cv=5, scoring=None, random_state=42, verbose=True, cat_cols=[], reg_name="",

                    preprocess=None):

    if scoring is None: 

        print("Pass scoring as evaluation function as an argument.")

        return None

    oof = []

    np.random.seed(random_state)

    

    seeds = np.random.randint(0, 100000, size=cv)

    

    for i in range(cv):

        k_fold = KFold(n_splits=cv, shuffle=True, random_state=seeds[i])

        temp = np.zeros(len(X))



        for fold, ids in enumerate(tqdm(k_fold.split(X, y), disable=not verbose)):

            if verbose: print("{} Fold.".format(fold+1))

            X_train, y_train = X[ids[0]], y[ids[0]]

            X_valid, y_valid = X[ids[1]], y[ids[1]]



    #         print(help(preprocess))

            if preprocess is not None:

#                 X_train, X_valid = partial(preprocess, train=X_train, test=X_valid)

                X_train, X_valid = preprocess(train=X_train, test=X_valid)



            if verbose: print("\tFitting...")

            if reg_name == "LightGBM":

                reg.fit(X_train, y_train, categorical_feature=cat_cols)

            elif reg_name == "CatBoost":

                reg.fit(X_train, y_train, cat_features=cat_cols)

            else:

                reg.fit(X_train, y_train)

            if verbose: print("\tPredicting...")

            p = reg.predict(X_valid)

            if verbose: print("\tEvaluating...")

            

            temp[ids[1]] += p

        oof = np.append(oof, scoring(y, temp))

        

    return np.mean(oof)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



def preprocess(train, test, target_col="G3", standardize=True, impute=True):

    work = None

    

    if not isinstance(train, np.ndarray):

        if target_col in train.columns:

            work = train[target_col]

            train = train.drop(target_col, axis=1)

    

    if impute:

        imputer = SimpleImputer(strategy="median")

        train = imputer.fit_transform(train)

        test = imputer.transform(test)

        

    if standardize:

        sc = StandardScaler()

        train = sc.fit_transform(train)

        test = sc.transform(test)

    

#     if work is not None:

#         train[target_col] = work.values

    return train, test
from sklearn.model_selection import cross_validate

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from xgboost import XGBRegressor, XGBRFRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.linear_model import ElasticNet, SGDRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor



X = df_train_.drop([target_col], axis=1)

y = df_train_[target_col]



X_other = X.drop(target_0_indx, axis=0).values

y_other = y.drop(target_0_indx, axis=0).values





d_regs = {

    "LightGBM": LGBMRegressor(),

    "CatBoost": CatBoostRegressor(verbose=False),

    "XGBoost": XGBRegressor(objective="reg:squarederror"),

    "XGRFBoost": XGBRFRegressor(objective="reg:squarederror"),

    "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1),

    "DecisionTree": DecisionTreeRegressor(),

    "Support Vector Machine": SVR(gamma="auto"),

    "ElasticNet": ElasticNet(),

    "SGD": SGDRegressor(),

#     "BaggingReg": BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, n_jobs=-1),

}



d_scores = {}

for reg_name, reg in d_regs.items():

    print("{} : ".format(reg_name), end="")

    

#     cross_validate(reg, X_other, y_other, cv=5, scoring=rmse_scorer)['test_score'].mean()

    d_scores[reg_name] = cross_validation(reg, X_other, y_other, cv=5, scoring=rmse, verbose=False, cat_cols=cat_cols, preprocess=preprocess,

                                         random_state=100)

    print(d_scores[reg_name])
plt.figure(figsize=(20, 10))

sort_scores = dict(sorted(d_scores.items(), key=lambda x: x[1]))

sns.barplot(list(sort_scores.keys()), list(sort_scores.values()))

plt.ylim(2.0)

plt.title("Crossvalidation Score")

plt.show()
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



smote = SMOTE()



X_0 = X.copy()

y_0 = y.copy()

y_0[y_0 != 0] = 10



k_fold = KFold(n_splits=5, shuffle=False)



list_cv = np.zeros(len(y_0))



for ids in k_fold.split(X_0, y_0):

    X_train, y_train = X_0.iloc[ids[0]], y_0.iloc[ids[0]]

    X_valid, y_valid = X_0.iloc[ids[1]], y_0.iloc[ids[1]]

    

    smote = SMOTE()

    X_train, y_train = smote.fit_sample(X_train, y_train)



    clf = LGBMClassifier()

    clf.fit(X_train, y_train)

    list_cv[ids[1]] += clf.predict_proba(X_valid)[:, 1]

    

roc_auc_score(y_0, list_cv)

# cross_validate(LGBMClassifier(), X_0, y_0, cv=5, scoring='roc_auc')['test_score'].mean()
# vis_model = CatBoostRegressor(verbose=False)

vis_model = LGBMRegressor()



df_normal = df_train_[df_train_[target_col] != 0]

X_normal = df_normal.drop(target_col, axis=1)

y_normal = df_normal[target_col]
vis_model.fit(X_normal, y_normal)

df_imp = pd.DataFrame([vis_model.feature_importances_, 

                      df_normal.drop(target_col, axis=1).columns],

                     index=['Importance', 'Feature']).T

df_imp = df_imp.sort_values('Importance', ascending=False)



plt.figure(figsize=(12, 24))

sns.barplot(x='Importance', y='Feature', data=df_imp)
X = df_train_.drop([target_col], axis=1)

y = df_train_[target_col]

X_other = X.drop(target_0_indx, axis=0)

y_other = y.drop(target_0_indx, axis=0)
import optuna

from sklearn.model_selection import train_test_split



def lgb_objective(trial):

    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal)

    max_depth = trial.suggest_int('max_depth', 1, 100)

    min_child_weight = trial.suggest_int('min_child_weight', 1, 100)

    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)

    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    

    model = LGBMRegressor(

        n_estimators=100,

        random_state=42,

#         n_estimators = n_estimators,

        max_depth = max_depth,

        min_child_weight = min_child_weight,

        subsample = subsample,

        colsample_bytree = colsample_bytree,

        boosting_type = boosting

    )

    model.fit(X_train, y_train)

    opt_pred = model.predict(X_test)

    return (rmse(y_test, opt_pred))
lgb_study = optuna.create_study()

lgb_study.optimize(lgb_objective, n_trials=150, n_jobs=-1)
from lightgbm import LGBMClassifier, LGBMRegressor



lgb_reg = LGBMRegressor(**lgb_study.best_params)

lgb_clf = LGBMClassifier()



X = df_train_.drop([target_col], axis=1)

y = df_train_[target_col]

X_test = df_test_.copy()
from imblearn.over_sampling import SMOTE



X_0 = X.copy()

y_0 = y.copy()

y_0[y_0 != 0] = 10



smote = SMOTE()

X_0, y_0 = smote.fit_sample(X_0, y_0)



lgb_clf.fit(X_0, y_0)



clf_predict = lgb_clf.predict(X_test)

predict_0_indx = X_test[clf_predict == 0].index
X_test_normal = X_test.drop(predict_0_indx, axis=0)

df_tmp = df_train_[df_train_[target_col] != 0]

X_ = df_tmp.drop([target_col], axis=1)

y_ = df_tmp[target_col]
lgb_reg.fit(X_, y_)

reg_predict = lgb_reg.predict(X_test_normal)
df_pred_reg = pd.DataFrame(reg_predict, index=X_test_normal.index, columns=[target_col])

df_pred_clf = pd.DataFrame(np.zeros(len(predict_0_indx)), index=predict_0_indx, columns=[target_col])



df_predict = pd.concat([df_pred_reg, df_pred_clf]).sort_index()

# df_predict = np.round(df_predict)

df_predict.to_csv('lgb_submit.csv')
from catboost import CatBoostClassifier, CatBoostRegressor



cat_reg = CatBoostRegressor(verbose=False)

cat_clf = CatBoostClassifier(verbose=False)



X = df_train_.drop([target_col], axis=1)

y = df_train_[target_col]

X_test = df_test_.copy()
from imblearn.over_sampling import SMOTE



X_0 = X.copy()

y_0 = y.copy()

y_0[y_0 != 0] = 10



smote = SMOTE()

X_0, y_0 = smote.fit_sample(X_0, y_0)



cat_clf.fit(X_0, y_0)



clf_predict = cat_clf.predict(X_test)

predict_0_indx = X_test[clf_predict == 0].index
X_test_normal = X_test.drop(predict_0_indx, axis=0)

df_tmp = df_train_[df_train_[target_col] != 0]

X_ = df_tmp.drop([target_col], axis=1)

y_ = df_tmp[target_col]
cat_reg.fit(X_, y_)

cat_predict = cat_reg.predict(X_test_normal)
df_pred_reg = pd.DataFrame(cat_predict, index=X_test_normal.index, columns=[target_col])

df_pred_clf = pd.DataFrame(np.zeros(len(predict_0_indx)), index=predict_0_indx, columns=[target_col])



df_predict = pd.concat([df_pred_reg, df_pred_clf]).sort_index()

# df_predict = np.round(df_predict)

df_predict.to_csv('cat_submit.csv')