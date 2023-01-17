# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
rawTrainingData = pd.read_csv('../input/train.csv')
inputData = rawTrainingData.drop(rawTrainingData.columns[rawTrainingData.columns.str.contains('Id',case = False)],axis = 1)
inputData = inputData.drop(inputData.columns[inputData.columns.str.contains('Cover_Type',case = False)],axis = 1)
outputData = rawTrainingData['Cover_Type']
clf = RandomForestClassifier(max_depth = 5)
X_train,X_test,Y_train,Y_test = train_test_split(inputData,outputData,test_size = 0.3)
clf.fit(X_train,Y_train)
predictionScore = clf.score(X_test,Y_test)
print('The prediction score for random forest classifier is ',predictionScore)
coefs = clf.feature_importances_
clf_grad = GradientBoostingClassifier(learning_rate = 0.1,loss='deviance',max_depth = 8,n_estimators=300)
clf_grad.fit(X_train,Y_train)
predictionScore_gradBoost = clf_grad.score(X_test,Y_test)
print('The prediction score for gradient boosting classifier is ',predictionScore_gradBoost)
rawTestingData = pd.read_csv('../input/test.csv')
inputTestingData = rawTestingData.drop(rawTestingData.columns[rawTestingData.columns.str.contains('Id',case = False)],axis = 1)
Y_test_predict = clf_grad.predict(inputTestingData)
submissionData = {'Id':rawTestingData['Id'],'Cover_Type':Y_test_predict}
submissionDF = pd.DataFrame.from_dict(submissionData)
columnOrder = ['Id','Cover_Type']
submissionDF = submissionDF.reindex(columns = columnOrder)
submissionDF.to_csv('Submission.csv',index = False)