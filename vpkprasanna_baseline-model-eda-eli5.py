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
from matplotlib import rcParams

from sklearn import model_selection

from sklearn import preprocessing

from sklearn import linear_model

from sklearn import metrics

import lightgbm as lgb

import catboost as cb 

import xgboost as xgb

import eli5

import plotly.graph_objects as go

import plotly.express as px
train = pd.read_csv("/kaggle/input/av-janatahack-machine-learning-in-agriculture/train_yaOffsB.csv")

test = pd.read_csv("/kaggle/input/av-janatahack-machine-learning-in-agriculture/test_pFkWwen.csv")

sub = pd.read_csv("/kaggle/input/av-janatahack-machine-learning-in-agriculture/sample_submission_O1oDc4H.csv")
train.head()
train.ID.nunique
rcParams["figure.figsize"] = 15,10

train.isna().sum().plot(kind="bar")
train.describe()
# value count for the target variable.

train.Crop_Damage.value_counts().plot(kind='bar',title='Crop Damaged',color=['red'])

train.groupby('Crop_Type')['Estimated_Insects_Count'].sum()
# Crop type which is more prone to pests

train.groupby('Crop_Type')['Estimated_Insects_Count'].sum().plot(kind='bar')
rcParams["figure.figsize"] = 10,10

train.groupby('Soil_Type')['Estimated_Insects_Count'].sum().plot(kind='bar')
fig = px.line(data_frame=train[:5000],x="ID",y="Estimated_Insects_Count")

fig.show()
pesti =  train["Pesticide_Use_Category"].value_counts()

pesti_df = pd.DataFrame({"pesticide":pesti.index,"frequency":pesti.values})
fig = px.bar(data_frame=pesti_df,x="pesticide",y="frequency",color = "pesticide")

fig.show()
# join test and train data

train['train_or_test']='train'

test['train_or_test']='test'

df=pd.concat([train,test])
df.head()
df['Number_Weeks_Used']=df['Number_Weeks_Used'].fillna(df['Number_Weeks_Used'].mode()[0])
df.head()


train=df.loc[df.train_or_test.isin(['train'])]

test=df.loc[df.train_or_test.isin(['test'])]

train.drop(columns={'train_or_test'},axis=1,inplace=True)

test.drop(columns={'train_or_test'},axis=1,inplace=True)
train.head()
train.isna().sum()
test.isna().sum()
train["kfold"] = -1

train = train.sample(frac=1).reset_index(drop=True)



y = train.Crop_Damage.values



kf =model_selection.StratifiedKFold(n_splits=5)



for f,(t_,v_) in enumerate(kf.split(X=train,y=y)):

    train.loc[v_,'kfold'] = f
train.head()
train = train.drop(["ID"],axis=1)

test = test.drop(["ID","Crop_Damage"],axis=1)
def run(train,fold):

    scores = []

#     lb = preprocessing.LabelEncoder()



    df_train = train[train.kfold !=fold].reset_index(drop=True)

    df_valid = train[train.kfold == fold].reset_index(drop=True)

    

    x_train = df_train.drop(["Crop_Damage"],axis=1)

#     x_train['ID']= lb.fit_transform(x_train['ID']) 

    y_train = df_train["Crop_Damage"].values

    

    x_valid = df_valid.drop(["Crop_Damage"],axis=1)

#     x_valid['ID']= lb.fit_transform(x_valid['ID']) 



    y_valid = df_valid["Crop_Damage"].values

    

    model = lgb.LGBMClassifier(random_state=27, max_depth=6, n_estimators=400)

    model.fit(x_train, y_train)

    preds = model.predict(x_valid)

    score = metrics.accuracy_score(y_valid, preds)

    print(f"Fold = {fold}, AUC = {score}")

for fold_ in range(5):

    run(train=train,fold=fold_)
def run(train,fold):

    scores = []

#     lb = preprocessing.LabelEncoder()

    df_train = train[train.kfold !=fold].reset_index(drop=True)

    df_valid = train[train.kfold == fold].reset_index(drop=True)

    

    x_train = df_train.drop(["Crop_Damage","kfold"],axis=1)

#     x_train['ID']= lb.fit_transform(x_train['ID']) 



    y_train = df_train["Crop_Damage"].values

    

    x_valid = df_valid.drop(["Crop_Damage","kfold"],axis=1)

#     x_valid['ID']= lb.fit_transform(x_valid['ID']) 

    y_valid = df_valid["Crop_Damage"].values

    

    model = xgb.XGBClassifier(objective = "multi:softprob",

              num_class = 3,

              max_depth = 8,

              eta = 0.01,

              subsample = 0.7,

              colsample_bytree = 0.8,

              min_child_weight = 40,

              max_delta_step = 3,

              gamma = 0.3,

              eval_metric = "merror",

                             )

    model.fit(x_train, y_train,)

    preds = model.predict(x_valid)

    score = metrics.accuracy_score(y_valid, preds)

    print(f"Fold = {fold}, AUC = {score}")

for fold_ in range(5):

    run(train=train,fold=fold_)
x_train = train.drop(["Crop_Damage","kfold"],axis=1)

y_train = train["Crop_Damage"].values
x_train.head()
model = xgb.XGBClassifier(objective = "multi:softprob",

              num_class = 3,

              max_depth = 8,

              eta = 0.01,

              subsample = 0.7,

              colsample_bytree = 0.8,

              min_child_weight = 40,

              max_delta_step = 3,

              gamma = 0.3,

              eval_metric = "merror")
model.fit(x_train,y_train)
y_pred_xgb = model.predict_proba(test)

y_pred_xgb
sub["Crop_Damage"] = y_pred_xgb

sub.head()

sub.to_csv("XGB_Model.csv")
eli5.explain_weights_xgboost(model,top=10)
eli5.show_weights(model,feature_names=x_train.columns.tolist())
eli5.show_prediction(model, x_train.iloc[1], feature_names = x_train.columns.tolist(), 

                show_feature_values=True)
def run(train,fold):

    scores = []

    df_train = train[train.kfold !=fold].reset_index(drop=True)

    df_valid = train[train.kfold == fold].reset_index(drop=True)

    

    x_train = df_train.drop(["Crop_Damage","kfold"],axis=1)

    y_train = df_train["Crop_Damage"].values

    

    x_valid = df_valid.drop(["Crop_Damage","kfold"],axis=1)

    y_valid = df_valid["Crop_Damage"].values

    

    model = cb.CatBoostClassifier(random_state=27, max_depth=4, task_type="CPU", devices="0:1", n_estimators=1000, verbose=500)

    model.fit(x_train, y_train)

    preds_t = model.predict(x_valid)

    score = metrics.accuracy_score(y_valid, preds_t)

    print(f"Fold = {fold}, AUC = {score}")
for fold_ in range(5):

    run(train=train,fold=fold_)
cat_model = cb.CatBoostClassifier(random_state=27, max_depth=4, task_type="CPU", devices="0:1", n_estimators=1000, verbose=500)

cat_model.fit(x_train,y_train)
y_pred_cat = cat_model.predict_proba(test)
y_pred_cat[:10]
predictions = list()

cb_weight=0.6 # Catboost

lb_weight=0.4 # LGBM

for i, j in zip(y_pred_cat, y_pred_xgb):

    xx = [(cb_weight * i[0]) + (lb_weight * j[0]),

          (cb_weight * i[1]) + (lb_weight * j[1]),

          (cb_weight * i[2]) + (lb_weight * j[2])]

    predictions.append(xx)

# print(predictions[:10])

preds_ensemble=np.argmax(predictions,axis=1)

preds_ensemble[:10]
sub["Crop_Damage"] = preds_ensemble

sub.to_csv("Ensemble.csv")

sub.head()