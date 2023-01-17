# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
feature = ["Pclass","Age","SibSp","Parch"]

oh_feature = ["Sex","Embarked"]

label = "Survived"
train[oh_feature].isnull().sum()
class Encode:

    def __init__(self,impute_values=None):

        self.impute_values = impute_values

        self.encodes = {}

    

    def onh(self, features,df):

        

        df_ens_lst = []

        for feature in features:

            ens = preprocessing.OneHotEncoder()

            feature_value = df[feature].fillna(self.impute_values[feature]).values.reshape(-1,1)

            ens.fit(feature_value)

            df_ens = ens.transform(feature_value)

            self.encodes[feature]=ens

            df_ens_lst.append(df_ens)

        return self.combine(df_ens_lst) 

        

    def combine(self,df_ens_lst):

        arr = np.hstack([d.toarray() for d in df_ens_lst])

        columns=np.concatenate([en.categories_[0] for en in self.encodes.values()])

        return pd.DataFrame(arr, columns=columns)

    

    def tranform(self,features,df):

        df_ens_lst = []

        for feature in features:

            ens = self.encodes[feature]

            feature_value = df[feature].fillna(self.impute_values[feature]).values.reshape(-1,1)

            df_ens = ens.transform(feature_value)

            df_ens_lst.append(df_ens)

        return self.combine(df_ens_lst) 
impute_values = {

    "Sex":"male",

    "Embarked" : "S"

}
enc = Encode(impute_values)
oh_featue_df = enc.onh(oh_feature, train)
X_train= pd.concat([oh_featue_df,train[feature]],sort=False,axis=1)
X_train
X_train["Age"] = train["Age"].fillna(0)
Rm = RandomForestClassifier(n_estimators=50)
Rm.fit(X_train.values,train[label])
Rm.score(X_train,train[label])
X_test = test[feature]
oh_feature_def = enc.tranform(oh_feature,test)
X_test = pd.concat([oh_feature_def,X_test],axis=1)
X_test.isnull().sum()
test["Survived"] = Rm.predict(X_test.fillna(0).values)
test[["PassengerId","Survived"]].to_csv("Random.csv",index=False)