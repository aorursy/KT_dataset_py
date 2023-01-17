# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")

fnc
sample_submission = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")

sample_submission.head()
loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")

loading.head()
train_scores = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")

train_scores.fillna(train_scores.mean(),inplace=True)
df_merge = fnc.merge(loading,on='Id')

df_merge
train = train_scores.merge(df_merge,on='Id').drop(['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2'],axis=1)

train
test = pd.DataFrame(sample_submission['Id'].apply(lambda x: int(x.split('_')[0]))).merge(df_merge,on='Id').drop_duplicates().reset_index(drop=True)

test
# from tensorflow import keras

# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Dense

# from sklearn.model_selection import KFold,cross_val_score



# def my_model():

#     model = Sequential()

#     model.add(Dense(1404,input_dim=1404,kernel_initializer='normal',activation='relu'))

#     model.add(Dense(702,kernel_initializer='normal',activation='relu'))

#     model.add(Dense(5,kernel_initializer='normal'))

    

#     model.compile(loss='mean_absolute_error',optimizer='adam')

#     return model



# # clf = keras.wrappers.scikit_learn.KerasRegressor(build_fn=my_model,epochs=50,batch_size=125,verbose=0)

# clf = my_model()

# clf.fit(train.iloc[:,1:],train_scores.iloc[:,1:],epochs=25,batch_size=100,validation_split=0.25,verbose=2)

# preds = clf.predict(test.iloc[:,1:])

# output = pd.DataFrame(preds)

# output.insert(0,'Id',test['Id'].apply(lambda x:str(x)))

# output.columns = train_scores.columns



# final = pd.DataFrame()

# for col in ['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']:

#     temp = pd.DataFrame(output['Id'].apply(lambda x:str(x)+ '_' + col))

#     temp['Predicted'] = output[col]

#     final = pd.concat([final,temp])

# final.head()
from sklearn.model_selection import train_test_split

import lightgbm as lgb



param = {'objective':'regression',

        'metric':'rmse',

        'bossting_type':'gbdt',

        'learning_rate':0.01,

        'max_depth':-1}



output = pd.DataFrame()



for target in ['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']:

    X_train,X_val,y_train,y_val = train_test_split(train.iloc[:,1:],train_scores[target],test_size=0.2,shuffle=True,random_state=20)

    train_data = lgb.Dataset(X_train,label=y_train)

    val_data = lgb.Dataset(X_val,label=y_val)

    

    bst = lgb.train(param,train_data,10000,early_stopping_rounds=15,valid_sets=[val_data],verbose_eval=-1)

    

    temp = pd.DataFrame(test['Id'].apply(lambda x:str(x)+ '_'+ target))

    temp['Predicted'] = bst.predict(test.iloc[:,1:])

    output = pd.concat([output,temp])

output = sample_submission.drop('Predicted',axis=1).merge(output,on='Id',how='left')

output.to_csv('sbumission.csv',index=False)

output