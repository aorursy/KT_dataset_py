# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
path='/kaggle/input/petfinder-adoption-prediction/'

# t=pd.read_json(path+'train_sentiment/25a834a2e.json', orient='split')

train_meta=pd.read_csv(path+'train/train.csv')

test_meta=pd.read_csv(path+'test/test.csv')

print(train_meta)

print(test_meta)
print(train_meta.info())

print(train_meta.describe())

train_meta.hist(figsize=(15,15))
train_meta.shape
train_meta.isna().sum()
print(train_meta.Name.unique())

print(len(train_meta.Name.unique()))
print(train_meta.Description.unique())

print(len(train_meta.Description.unique()))
# count the number of duplicate values

from collections import Counter

c = Counter(list(zip(train_meta.columns)))

c
import matplotlib.pyplot as plt

f = plt.figure(figsize=(19, 15))

plt.matshow(train_meta.corr(), fignum=f.number)

plt.xticks(range(train_meta.shape[1]), train_meta.columns, fontsize=14, rotation=45)

plt.yticks(range(train_meta.shape[1]), train_meta.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);



t=train_meta.drop(["AdoptionSpeed",'Name','RescuerID','Description','PetID'], axis=1).apply(lambda x: x.corr(train_meta.AdoptionSpeed))

t
#Linear Regression?



# PCA to see the data in 2 components

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn import preprocessing



x = train_meta.drop(["AdoptionSpeed",'Name','RescuerID','Description','PetID'], axis=1).values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled)



pca = PCA(n_components=2)

principalComponents=pca.fit_transform(df)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, train_meta[['AdoptionSpeed']]], axis = 1)

finalDf
finalDf.info()
finalDf.drop('AdoptionSpeed',axis=1).plot(figsize=(18,5))

df.isnull().values.any()

# The “False” output confirms that there are no null values in the dataframe.
pca = PCA(n_components=1)

principalComponents=pca.fit_transform(train_meta.drop(["AdoptionSpeed",'Name','RescuerID','Description','PetID'], axis=1))

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1'])

finalDf = pd.concat([principalDf, train_meta[['AdoptionSpeed']]], axis = 1)

plt.scatter(finalDf['principal component 1']

               , finalDf['AdoptionSpeed']

               , c = 'r'

               , s = 50)

plt.show()
import pandas as pd

import numpy as np

from scipy import stats

from datetime import datetime

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline



# Validate linear relationship

plt.scatter(finalDf['principal component 1'], finalDf['AdoptionSpeed'])
X = pd.DataFrame(finalDf[['principal component 1','principal component 2']])

y = pd.DataFrame(finalDf['AdoptionSpeed'])

model = LinearRegression()

scores = []

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X, y)):

 model.fit(X.iloc[train,:], y.iloc[train,:])

 score = model.score(X.iloc[test,:], y.iloc[test,:])

 scores.append(score)

print(scores)
train_meta
# https://acadgild.com/blog/logistic-regression-multiclass-classification

# train and validate

# then apply for test

from sklearn.datasets import fetch_mldata

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import pandas as pd

import numpy as np



X = pd.DataFrame(finalDf[['principal component 1','principal component 2']])

y = pd.DataFrame(finalDf['AdoptionSpeed'])



# valid_size: what proportion of original data is used for test set

train_img, valid_img, train_lbl, valid_lbl = train_test_split(X,y,test_size=0.2, random_state=122)

    

#Fit the model

model = LogisticRegression(solver = 'lbfgs')

model.fit(train_img, train_lbl)

    

#Validate the fitting

# use the model to make predictions with the valid data

y_pred = model.predict(valid_img)

print(y_pred)

print(valid_lbl)

# how did our model perform?

count_misclassified = (valid_lbl['AdoptionSpeed'] != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(valid_lbl, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))

# valid_size: what proportion of original data is used for test set

train_img, valid_img, train_lbl, valid_lbl = train_test_split(train_meta.drop(["AdoptionSpeed",'Name','RescuerID','Description','PetID'], axis=1),

     train_meta["AdoptionSpeed"],test_size=0.2, random_state=122)



#Standardize

scaler = StandardScaler()

# Fit on training set only.

scaler.fit(train_img)

# Apply transform to both the training set and the test set.

train_img = scaler.transform(train_img)

test_img = scaler.transform(valid_img)

    

#Fit the model

model = LogisticRegression(solver = 'lbfgs')

model.fit(train_img, train_lbl)

    

#Validate the fitting

# use the model to make predictions with the test data

y_pred = model.predict(valid_img)

# how did our model perform?

count_misclassified = (valid_lbl != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(valid_lbl, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
test_meta['PetID']
#Predict for test data the fitting

# use the model to make predictions with the test data

y_pred = model.predict(test_meta.drop(['Name','RescuerID','Description','PetID'], axis=1))

y_pred=pd.DataFrame(data = y_pred

             , columns = ['AdoptionSpeed'])

submission= pd.concat([test_meta['PetID'],y_pred], axis = 1)

submission.to_csv('/kaggle/working/'+'submission.csv')

submission