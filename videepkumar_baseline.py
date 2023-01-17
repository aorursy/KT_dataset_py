import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression

from sklearn.impute import KNNImputer

from sklearn.metrics import roc_auc_score,accuracy_score

from xgboost import XGBClassifier 

#IMPORT TRAIN DATA

df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")

df_train.head()
#IMPORT TEST DATA

df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")

df_test.head()
Y=df_train["y"]

Y=Y.to_numpy()

X=df_train.loc[:, df_train.columns != 'y']

X.nunique()

combo=pd.concat(objs=[X,df_test])

combo=pd.get_dummies(data=combo,columns=["x9","x16","x17","x18","x19"],dummy_na=True,drop_first=True)

X_train_dummy=pd.DataFrame(data=combo[0:Y.shape[0]])

X_test_dummy=pd.DataFrame(data=combo[Y.shape[0]:])

X_test_dummy.head()

X_train_normalized = scale(X_train_dummy)

X_test_normalized=scale(X_test_dummy)

X_train_try=pd.DataFrame(data=X_train_normalized)

X_train_try.head()

X_train_filled=KNNImputer(n_neighbors=5).fit_transform(X_train_normalized)

X_train_filled=pd.DataFrame(data=X_train_filled)

X_train_filled.columns=X_train_dummy.columns

X_train_filled.head()

X_test_filled=KNNImputer(n_neighbors=5).fit_transform(X_test_normalized)

X_test_filled=pd.DataFrame(data=X_test_filled)

X_test_filled.columns=X_test_dummy.columns

X_test_filled.head()
X_train=X_train_filled

X_test=X_test_filled

xgb=XGBClassifier(n_estimators=500,subsample=1,colsample_bytree=1,max_depth=6,gamma=0,random_state=2)

xgb.fit(X_train,Y)

pred=xgb.predict_proba(X_test)

out=pd.DataFrame()

out["Id"]=range(0,4000)

out["Predicted"]=pd.DataFrame(pred[:,1])

out.head()
out.to_csv('results_out.csv', index=False)