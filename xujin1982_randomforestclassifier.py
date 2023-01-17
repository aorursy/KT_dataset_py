#Load all the libraries we will be using

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Get the training file and testing file

df=pd.read_json("../input/train.json")

df.reset_index(inplace=True,drop=True)



df_test=pd.read_json("../input/test.json")

df_test.reset_index(inplace=True,drop=True)
df.isnull().any()
df["num_photos"]=df["photos"].apply(np.size,axis=0)

df["num_features"]=df["features"].apply(np.size,axis=0)



#Also lets change the interest level through numeric mapping



df["interest_level"].replace({'low': 0, 'medium': 1,'high':2},inplace=True)

df.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score



X=df[["bathrooms","bedrooms","latitude","listing_id","longitude","price","num_photos"

      ,"num_features"]].values

Y=df["interest_level"].values



clf=RandomForestClassifier()

clf=clf.fit(X,Y)

fi=clf.feature_importances_



score = cross_val_score(clf, X, Y, cv=10)

#metrics.accuracy_score(predicted) 

np.mean(score)