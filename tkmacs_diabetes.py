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
train = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/train.csv').drop(['Patient number'],axis=1)

train['m']=((train['Gender']=="male")&(train['Waist'] > 33))|((train['Gender']=="female")&(train['Waist']>35))

train['BP'] = train['Systolic BP']-train['Diastolic BP']

train['normal']=((train['Height']*0.0254)**2)*22 - train['Weight']*0.453592
import xgboost as xgb

model = xgb.XGBClassifier()
train['Gender'] = train['Gender'].map({'male':1,'female':0})

X=train.drop(['Diabetes'],axis=1).values

Y=train['Diabetes'].values

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.model_selection import StratifiedKFold



scoring = ['roc_auc']



# 分割の仕方

cv = StratifiedKFold(n_splits=3, random_state=0)



# Cross Validation

scores = cross_validate(model, X, Y, scoring=scoring, cv=cv, return_train_score=False)





print(scores['test_roc_auc'])



from imblearn.over_sampling import SMOTE



# SMOTE

smote = SMOTE(random_state=71)

X_train_resampled, y_train_resampled = smote.fit_sample(train, Y)

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.model_selection import StratifiedKFold



scoring = ['roc_auc']



# 分割の仕方

cv = StratifiedKFold(n_splits=3, random_state=0)



# Cross Validation

scores = cross_validate(model, X_train_resampled, y_train_resampled, scoring=scoring, cv=cv, return_train_score=False)





print(scores['test_roc_auc'])

hist = model.fit(X_train_resampled,y_train_resampled)
from matplotlib import pyplot as plt

_, ax = plt.subplots(figsize=(12, 4))

xgb.plot_importance(hist,

                    ax=ax,

                    importance_type='weight',

                    show_values=True)

plt.show()
test = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/test.csv').drop(['Patient number'],axis=1)

test['m']=((test['Gender']=="male")&(test['Waist'] > 33))|((test['Gender']=="female")&(test['Waist']>35))

test['BP'] = test['Systolic BP']-test['Diastolic BP']

test['normal']=((test['Height']*0.0254)**2)*22 - test['Weight']*0.453592

test['Gender'] = test['Gender'].map({'male':1,'female':0})
p = model.predict_proba(test.values)[:,1]
sample = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv',index_col = 0)

sample['Diabetes'] = p
sample.to_csv('predict_xgb.csv',header = True)
import lightgbm as lgb

model=lgb.LGBMClassifier()
from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.model_selection import StratifiedKFold



scoring = ['roc_auc']



# 分割の仕方

cv = StratifiedKFold(n_splits=3, random_state=0)



# Cross Validation

scores = cross_validate(model, X, Y, scoring=scoring, cv=cv, return_train_score=False)





print(scores['test_roc_auc'])



from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.model_selection import StratifiedKFold



scoring = ['roc_auc']



# 分割の仕方

cv = StratifiedKFold(n_splits=3, random_state=0)



# Cross Validation

scores = cross_validate(model, X_train_resampled, y_train_resampled, scoring=scoring, cv=cv, return_train_score=False)





print(scores['test_roc_auc'])

hist = model.fit(X_train_resampled,y_train_resampled)



imp_df = pd.DataFrame()

imp_df["feature"] = train.drop(['Diabetes'],axis=1).columns

imp_df["importance"] = hist.feature_importances_

imp_df = imp_df.sort_values("importance")



# 可視化

plt.figure(figsize=(7, 10))

plt.barh(imp_df.feature, imp_df.importance)

plt.xlabel("Feature Importance")

plt.show()
def get_feature_importances(X, y, shuffle=False):

    # 必要ならば目的変数をシャッフル

    if shuffle:

        y = np.random.permutation(y)



    # モデルの学習

    clf = xgb.XGBClassifier(random_state=42)

    clf.fit(X, y)



    # 特徴量の重要度を含むデータフレームを作成

    imp_df = pd.DataFrame()

    imp_df["feature"] = X.columns

    imp_df["importance"] = clf.feature_importances_

    return imp_df.sort_values("importance", ascending=False)



# 実際の目的変数でモデルを学習し、特徴量の重要度を含むデータフレームを作成

actual_imp_df = get_feature_importances(X_train_resampled.drop(['Diabetes'],axis=1), y_train_resampled, shuffle=False)



# 目的変数をシャッフルした状態でモデルを学習し、特徴量の重要度を含むデータフレームを作成

N_RUNS = 100

null_imp_df = pd.DataFrame()

for i in range(N_RUNS):

    imp_df = get_feature_importances(X_train_resampled.drop(['Diabetes'],axis=1), y_train_resampled, shuffle=True)

    imp_df["run"] = i + 1

    null_imp_df = pd.concat([null_imp_df, imp_df])
def display_distributions(actual_imp_df, null_imp_df, feature):

    # ある特徴量に対する重要度を取得

    actual_imp = actual_imp_df.query(f"feature == '{feature}'")["importance"].mean()

    null_imp = null_imp_df.query(f"feature == '{feature}'")["importance"]



    # 可視化

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    a = ax.hist(null_imp, label="Null importances")

    ax.vlines(x=actual_imp, ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')

    ax.legend(loc="upper right")

    ax.set_title(f"Importance of {feature.upper()}", fontweight='bold')

    plt.xlabel(f"Null Importance Distribution for {feature.upper()}")

    plt.ylabel("Importance")

    plt.show()



# 実データにおいて特徴量の重要度が高かった上位5位を表示

for feature in actual_imp_df["feature"][:10]:

    display_distributions(actual_imp_df, null_imp_df, feature)
THRESHOLD = 80



# 閾値を超える特徴量を取得

imp_features = []

for feature in actual_imp_df["feature"]:

    actual_value = actual_imp_df.query(f"feature=='{feature}'")["importance"].values

    null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values

    percentage = (null_value < actual_value).sum() / null_value.size * 100

    if percentage >= THRESHOLD:

        imp_features.append(feature)



imp_features
model=xgb.XGBClassifier()

model.fit(X_train_resampled[imp_features],y_train_resampled)
p = model.predict_proba(test[imp_features])[:,1]

sample = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv',index_col = 0)

sample['Diabetes'] = p

sample.to_csv('predict_lgbm.csv',header = True)
from sklearn.decomposition import PCA  

import cufflinks as cf                       # おしゃれな可視化のために必要なライブラリ その2

cf.go_offline()

pca = PCA(n_components=10)                     # 3次元に圧縮するPCAインスタンスを作成

X_pca = pca.fit_transform(X) 
embed3 = pd.DataFrame(X_pca)                      # 可視化のためにデータフレームに変換

embed3["Diabetes"] = train["Diabetes"]

#embed3 = embed3.rename(columns={0: '0',1:'1',2:'2'})

embed3.head()
plt.figure(figsize=(6, 6))

plt.scatter(embed3[8], embed3[2], alpha=0.8, c=list(embed3['Diabetes']))

plt.grid()

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.show()

import pyclustering

from pyclustering.cluster import xmeans
X_train_resampled
X_train_resampled[imp_features]
actual_imp_df