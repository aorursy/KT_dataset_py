import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression

from sklearn.impute import KNNImputer

from sklearn.metrics import roc_auc_score,accuracy_score

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
#IMPORT TRAIN DATA

df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")

df_train.head()
df_train.describe()
#IMPORT TEST DATA

df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")

df_test.head()
df_test.describe()
df_train.isnull().sum()
df_test.isnull().sum()
Y_train=df_train['y']

Y_train=Y_train.to_numpy()

print(Y_train.shape)
X_test=df_test

X_test.shape
X_train=df_train.loc[:, df_train.columns !='y']

X_test=df_test.loc[:,df_test.columns !='y']

combo=pd.concat(objs=[X_train,X_test])

combo.describe()
combo.nunique()
combo=pd.get_dummies(data=combo,columns=["x9","x16","x17","x18","x19"],dummy_na=True,drop_first=True)

combo.head()
combo.describe()
X_train_dummy=pd.DataFrame(data=combo[0:Y_train.shape[0]])

X_train_dummy.head()
X_test_dummy=pd.DataFrame(data=combo[Y_train.shape[0]:])

X_test_dummy.head()
X_train_normalized = scale(X_train_dummy)

X_test_normalized=scale(X_test_dummy)

X_train_try=pd.DataFrame(data=X_train_normalized)

X_train_try.head()
X_train_filled=KNNImputer(n_neighbors=5).fit_transform(X_train_normalized)

X_train_filled=pd.DataFrame(data=X_train_filled)

X_train_filled.columns=X_train_dummy.columns

X_train_filled.head()
X_train_filled.isnull().sum()
X_test_filled=KNNImputer(n_neighbors=5).fit_transform(X_test_normalized)

X_test_filled=pd.DataFrame(data=X_test_filled)

X_test_filled.columns=X_test_dummy.columns

X_test_filled.head()
X_test_filled.isnull().sum()
X_train=X_train_filled

X_test=X_test_filled
svmc=SVC(random_state=2)

svmc.fit(X_train,Y_train)
predsvm=svmc.predict(X_test)
predsvm
rfc=RandomForestClassifier(n_estimators=500,random_state=2)

rfc.fit(X_train,Y_train)
predrfc=rfc.predict(X_test)
predrfc
xgb=XGBClassifier(n_estimator=500,random_state=2)

xgb.fit(X_train,Y_train)
predxgb=xgb.predict(X_test)

predxgb
predxgb.shape
predf=predsvm+predxgb+predrfc

predf=predf/3
data=pd.read_csv("/kaggle/input/isteml2020/sampleSubmission.csv")

data.head()
predxgb[7]
for i in range(len(predf)):

    data.iloc[i:,1]=predf[i]

    
data
for i in range(4000):

    print(predf[i])
data.to_csv('results_out.csv',index=False)