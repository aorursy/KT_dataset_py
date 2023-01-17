



%matplotlib inline



importances=xgb_model.feature_importances_



importance_frame = pd.DataFrame({'Importance': list(importances), 'Feature': list(X_train.keys())})

importance_frame.sort_values(by = 'Importance', inplace = True)

importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn import svm

from sklearn import neural_network

from sklearn.model_selection import cross_val_predict

from sklearn import ensemble

from sklearn.metrics import roc_curve, auc





data = pd.read_csv("./input/maintenance_data.csv")

print (data.head(2))

print (list(data))





numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#data.select_dtypes(include=numerics).diff().hist()

#plt.show()

#preprocessing:

inputCat = data[["team","provider"]]

inputCatDummy = pd.get_dummies(inputCat)



inputCont = data[["pressureInd","moistureInd","temperatureInd","lifetime"]]

input = pd.concat([inputCont,inputCatDummy],axis=1)

output = data[["broken"]]



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

print (input.head(2))



new_df=pd.concat([input,output],axis=1)



from sklearn.utils import resample



# Separate majority and minority classes

df_majority = new_df[new_df.broken==0]

df_minority = new_df[new_df.broken==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=603,    # to match majority class

                                 random_state=123) # reproducible results

 

# Combine majority class with upsampled minority class

balanced_df = pd.concat([df_majority, df_minority_upsampled])



# Display new class counts

print (balanced_df.broken.value_counts())



from sklearn.model_selection import train_test_split



y=balanced_df['broken']

X=balanced_df.drop('broken', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



# create training and testing vars

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)



import xgboost as xgb



from xgboost import XGBClassifier



xgb_model = XGBClassifier(learning_rate=0.08, n_estimators=200, max_depth=10)

xgb_model.fit(X_train, y_train)



print (xgb_model.score(X_train,y_train))



from sklearn.metrics import accuracy_score

pred_y_0 = xgb_model.predict(X_test)

print( accuracy_score(pred_y_0, y_test))



%matplotlib inline



importances=xgb_model.feature_importances_



importance_frame = pd.DataFrame({'Importance': list(importances), 'Feature': list(X_train.keys())})

importance_frame.sort_values(by = 'Importance', inplace = True)

importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')