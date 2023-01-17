# import necessary package for this attempt

import pandas as pd

import numpy as np

import matplotlib as plt

% matplotlib inline
#Input data files are available in the "../input/" directory.

train_df = pd.read_csv("../input/train.csv")

test_df = pd. read_csv ("../input/test.csv")
corr_df = train_df.corr()
# From correlation coefficient, Pcklass is the most negative impact, and Fare is the the most postive impact factor

corr_df.sort_values('Age',axis=0,ascending = True)
# run PCA 

from sklearn.decomposition import PCA

train_df.info()
pca = PCA(n_components=5)

train_df_t=train_df.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','Parch'],axis=1)

train_df_t.head()
pca.fit(train_df_t)