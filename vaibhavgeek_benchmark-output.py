import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# this imports the required libraries
# print(os.listdir("../input/testacm1/"))
df_train = pd.read_csv('../input/svnit-ml-1/train.csv')
df_test = pd.read_csv('../input/testacm1/test.csv')
# read the csv file as pandas dataframes
df_train.head()
train_X = df_train.loc[:, 'X1':'X23']
train_y = df_train.loc[:, 'Y']
#what is input and what is the output?
rf = RandomForestClassifier(n_estimators=50, random_state=123)
# what is random forest classifier and how it works can be found over here with proper description,
# https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2034%20Random_Forest.md
# multiple resources given for random forest classifier in the overview of this competition
rf.fit(train_X, train_y)
# train it with the data
df_test = df_test.loc[:, 'X1':'X23']
pred = rf.predict_proba(df_test)
#prediction dataframe, know more about dataframes and operations on it from the pandas documentation
result = pd.DataFrame(pred[:,1])
result.columns = ['Y']
result.index.name = 'ID'
result.index += 1 
result.to_csv('output.csv')
# write to csv file 
