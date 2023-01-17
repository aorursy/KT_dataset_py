import pandas as pd

import numpy as np

import os

import joblib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings('ignore')
_input_path = "../input/1056lab-diabetes-diagnosis/"

df_train = pd.read_csv(_input_path + "train.csv", index_col=0)

df_test = pd.read_csv(_input_path + "test.csv", index_col=0)

target_col = "Diabetes"
df_train.head()
df_train.groupby(["Gender", target_col]).std()
df_train.groupby(["Gender", target_col]).max()
df_test.head()
sns.countplot(df_train[target_col])

plt.show()
df_train.columns
from sklearn.preprocessing import LabelEncoder



obj_cols = []

for col in df_test.columns:

    

    if df_train[col].dtype == "object": 

        obj_cols.append(col)

        continue

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    sns.distplot(df_train[col], kde=False, label="train", ax=ax1)

    sns.distplot(df_test[col], kde=False, label="test", ax=ax2)

print("Object Columns : {}".format(obj_cols))
sns.distplot(df_test[df_test["Chol/HDL ratio"]<12]["Chol/HDL ratio"], kde=False)

plt.show()
for col in obj_cols:

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    

    order = df_train[col].unique()

    sns.countplot(df_train[col], ax=ax1, label="train", order=order)

    sns.countplot(df_test[col], ax=ax2, label="test", order=order)
sns.heatmap(df_train.corr(), square=True, vmax=1, vmin=-1, center=0)

plt.show()



df_train.corr()[target_col].sort_values()
plt.figure(figsize=(18,9))

sns.heatmap(df_train.isnull(), cbar=False)
df_test.columns
df_train.head()
df_train.groupby(["Gender"])
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 6))

df_train[df_train[target_col] == 1]['Age'].apply(np.log).plot(kind='hist', bins=100, title='Log col name - 1', color='#348ABD', xlim=(-3, 10), ax=ax1)

df_train[df_train[target_col] == 0]['Age'].apply(np.log).plot(kind='hist', bins=100, title='Log col name - 0', color='#348ABD', xlim=(-3, 10), ax=ax2)

plt.show()
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder, StandardScaler



def prepocess(X_train, X_valid=None, y_train=None, random_state=42, return_df=True):

    smote = SMOTE(random_state=random_state)

    le = LabelEncoder()

    sc = StandardScaler()

    

    if X_valid is None:

        print("X_valid is None")

        X_valid = X_train.copy()

    

    # Age over 40

    X_train["age_over_40"] = (X_train["Age"] > 40).values

    X_valid["age_over_40"] = (X_valid["Age"] > 40).values

    

    # round Age

    X_train["round_age"] = np.round(df_train["Age"] * 0.1)

    X_valid["round_age"] = np.round(X_valid["Age"] * 0.1)

    

    # コレステロールの基準値

    X_train["chol_normal"] = X_train["Chol/HDL ratio"] > 5

    X_valid["chol_normal"] = X_valid["Chol/HDL ratio"] > 5

    

    # BPの幅

    X_train["BP_width"] = X_train["Systolic BP"] - X_train["Diastolic BP"]

    X_valid["BP_width"] = X_valid["Systolic BP"] - X_valid["Diastolic BP"]

    

    obj_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    for col in obj_cols:        

        X_train[col] = le.fit_transform(X_train[col])

        X_valid[col] = le.transform(X_valid[col])

    

    X_train_sampled, y_train_sampled = smote.fit_sample(X_train, y_train)

    

    cols = X_train_sampled.columns

    X_train_sampled = sc.fit_transform(X_train_sampled)

    X_valid = sc.transform(X_valid)

    

    if return_df:

        X_train_sampled = pd.DataFrame(X_train_sampled, columns=cols)

        X_valid = pd.DataFrame(X_valid, columns=cols)

    

    return X_train_sampled, X_valid, y_train_sampled
import lightgbm as lgb



# X = df_train.drop(["Gender", target_col], axis=1)

X = df_train.drop([target_col], axis=1)

y = df_train[target_col]



X, _, y = prepocess(X, y_train=y)



clf = lgb.LGBMClassifier()

clf.fit(X, y)



df_imp = pd.DataFrame([clf.feature_importances_, X.columns],

                     index=['Importance', 'Feature']).T

df_imp = df_imp.sort_values('Importance', ascending=False)



plt.figure(figsize=(12, 24))

sns.barplot(x='Importance', y='Feature', data=df_imp)
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, accuracy_score

from tensorflow.keras.callbacks import EarlyStopping



def crossvalidation(clf, X, y, cv=5, scoring="auc", preprocessing=None, random_state=42,

                   keras=False):

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    

    oof = np.zeros(len(X))

    for i, ids in enumerate(skf.split(X, y)):

        X_train, y_train = X.iloc[ids[0]], y.iloc[ids[0]]

        X_valid, y_valid = X.iloc[ids[1]], y.iloc[ids[1]]

        

        if preprocessing is not None:

            X_train, X_valid, y_train = preprocessing(X_train, X_valid, y_train)

        

        clf.fit(X_train, y_train)



        if scoring == "auc":

            oof[ids[1]] = clf.predict_proba(X_valid)[:, 1]

        else:

            oof[ids[1]] = clf.predict(X_valid)

        

    # いつか追加する

    if scoring == "auc":

        fun_score = roc_auc_score

    elif scoring == "acc" or scoring == "accuracy":

        fun_score = accuracy_score

    

    return fun_score(y, oof)
from sklearn.svm import SVC

import lightgbm as lgb

import xgboost as xgb

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier



X = df_train.drop([target_col], axis=1)

y = df_train[target_col]



dict_clfs = {

    "SVM": SVC(probability=True, class_weight="balanced"),

    "LGB": lgb.LGBMClassifier(class_weight="balanced"),

    "XGB": xgb.XGBClassifier(),

    "LR": LogisticRegression(max_iter=10000),

#     "SGD": SGDClassifier(loss="hinge"),

    "Ada": AdaBoostClassifier(),

    "Extra": ExtraTreesClassifier(),

#     "MLP": MLPClassifier(max_iter=10000)

}



dict_oof = {}

for clf_name, clf in dict_clfs.items():

    dict_oof[clf_name] = crossvalidation(clf, X, y, preprocessing=prepocess, cv=10)

    print("{} : {}".format(clf_name, dict_oof[clf_name]))

df_oof = pd.DataFrame(dict_oof.values(), index=dict_oof.keys(), columns=["Score"]).sort_values(by="Score", ascending=False)
y_under = round(df_oof["Score"].min() * 0.95, 2)

y_over = round(df_oof["Score"].max() * 1.05, 2)



sns.barplot(x=df_oof.index, y=df_oof["Score"])

plt.ylim(y_under, y_over)

plt.title("Crossvalidation\nAUC")

plt.xlabel("Method")

plt.show()
import optuna

from sklearn.model_selection import train_test_split
def lgb_objective(trial):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    X_train, X_valid, y_train = prepocess(X_train, X_valid, y_train)

    dtrain = lgb.Dataset(X_train, label=y_train)

    

    params = {

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        'boosting': trial.suggest_categorical('boosting', ["gbdt", "dart"])

    }

    

#     clf = lgb.train(params, dtrain)

    clf = lgb.LGBMClassifier(**params).fit(X_train, y_train)

    pred = clf.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, pred)

    return 1 - score
X = df_train.drop(target_col, axis=1)

y = df_train[target_col]



lgb_study = optuna.create_study()

lgb_study.optimize(lgb_objective, n_trials=100, n_jobs=-1)
study.best_params
X = df_train.drop(target_col, axis=1)

y = df_train[target_col]



clf = lgb.LGBMClassifier(**lgb_study.best_params)

crossvalidation(clf, X, y, preprocessing=prepocess, cv=10)
# clf = dict_clfs[df_oof.index[0]]

clf = lgb.LGBMClassifier(**lgb_study.best_params)

X = df_train.drop(target_col, axis=1)

y = df_train[target_col]

X_test = df_test.copy()
X_, X_test_, y_ = prepocess(X, X_test, y)
clf.fit(X_, y_)

predict = clf.predict_proba(X_test_)[:, 1]
df_submit = pd.read_csv(_input_path + "sampleSubmission.csv")

df_submit[target_col] = predict

df_submit.to_csv("LGB_submit.csv", index=False)
clf = SVC(probability=True)

X = df_train.drop(target_col, axis=1)

y = df_train[target_col]

X_test = df_test.copy()
X_, X_test_, y_ = prepocess(X, X_test, y)
clf.fit(X_, y_)

predict = clf.predict_proba(X_test_)[:, 1]
df_submit = pd.read_csv(_input_path + "sampleSubmission.csv")

df_submit[target_col] = predict

df_submit.to_csv("SVM_submit.csv", index=False)
clf = LogisticRegression()

X = df_train.drop(target_col, axis=1)

y = df_train[target_col]

X_test = df_test.copy()



X_, X_test_, y_ = prepocess(X, X_test, y)



clf.fit(X_, y_)

predict = clf.predict_proba(X_test_)[:, 1]



df_submit = pd.read_csv(_input_path + "sampleSubmission.csv")

df_submit[target_col] = predict

df_submit.to_csv("LR_submit.csv", index=False)