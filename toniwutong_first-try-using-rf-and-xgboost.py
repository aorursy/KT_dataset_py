# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
full = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

full.head()
full.isnull().sum()

full=full.replace({"Attrition": {"Yes":1, "No":0}})
full=full.drop("Over18",axis=1)

full=full.replace({

    "OverTime": {"Yes":1, "No":0}

})

full=full.drop("StandardHours",axis=1)

full["Male"]=full["Gender"].map(lambda x: 1 if x=="Male" else 0)

full=full.drop("Gender",axis=1)
categorical_features = full.select_dtypes(include = ["object"]).columns

numerical_features = full.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("Attrition")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))
full_num = full[numerical_features]

full_cat = full[categorical_features]
from scipy.stats import skew

skewness = full_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index

full_num[skewed_features] = np.log1p(full_num[skewed_features])
full_cat = pd.get_dummies(full_cat)

#

X=pd.concat([full_num,full_cat],axis=1)

print("New Features:"+str(X.shape[1]))

Y=full.Attrition
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

print("X_train : " + str(X_train.shape))

print("X_test : " + str(X_test.shape))

print("y_train : " + str(y_train.shape))

print("y_test : " + str(y_test.shape))

# Standardize numerical features

from sklearn.preprocessing import StandardScaler

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
#randomForest

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,max_features=7)

rf.fit(X_train,y_train)

rf.score(X_train,y_train)

rf.score(X_test,y_test)

Imp=rf.feature_importances_

Importance=pd.DataFrame(Imp,index=X_train.columns,columns=["Importance"])

Importance=Importance.sort_values(by="Importance",ascending=False)

sns.set(font_scale=.5)

sns.barplot(y=Importance.index,x=Importance.Importance)
import xgboost as xgb
dtrain=xgb.DMatrix(X_train,label=y_train)

dtest=xgb.DMatrix(X_test,label=y_test)

param={'max_depth':2,'eta':0.01,'subsampe':0.5,'objective':'binary:logistic','booster':'gbtree'}

watchlist=[(dtest,'eval'),(dtrain,'train')]

num_round=3000

bst=xgb.train(param,dtrain,num_round,watchlist)
y_test_pred=bst.predict(dtest)

y_test_pred[y_test_pred>0.5]=1

y_test_pred[y_test_pred<=0.5]=0

y_train_pred=bst.predict(dtrain)

y_train_pred[y_train_pred>0.5]=1

y_train_pred[y_train_pred<=0.5]=0

print("Accurary on Training set :", sum(y_train_pred==y_train)/len(y_train))

print("Accurary on Test set :",sum(y_test_pred==y_test)/len(y_test))