import pandas as pd

import xgboost as xgb

from sklearn.cross_validation import  train_test_split

from sklearn.metrics import accuracy_score, log_loss

import numpy as np

from sklearn.preprocessing import  LabelEncoder



df_train = pd.read_csv('../input/train.csv',sep=',')



le = LabelEncoder()



df_train_wihtout_class = df_train.drop(['species', 'id'], axis=1)



df_train_wihtout_class = df_train_wihtout_class.fillna(0)

lebel = le.fit_transform(df_train['species'])



for columnsname in list(df_train_wihtout_class):

    df_train_wihtout_class[columnsname] = df_train_wihtout_class[columnsname].astype(np.float)



train_x,test_x,train_y,test_y = train_test_split(df_train_wihtout_class,lebel)



bst = xgb.XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 nthread=4,

 scale_pos_weight=1,

 seed=27,

 objective='multi:softprob')

bst.fit(train_x,train_y)



def acc_analysis():

    ypred = bst.predict(test_x)

    acc = accuracy_score(ypred,test_y)

    print(acc)



acc_analysis()