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



df["interest_level"].replace({'low': 0, 'medium': 2,'high':3},inplace=True)

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
plt.figure(figsize=(9,5))

plt.barh((np.arange(X.shape[1])),fi,align='center')

plt.yticks(np.arange(X.shape[1]), ("bathrooms","bedrooms","latitude","listing_id","longitude","price","num_photos"

      ,"num_features"))

plt.xlabel('Importance Score')

plt.ylabel('Features')

plt.title('Feature Importance measure using Random Forests')

plt.show()
#making a flat array out of the column

import re

des=np.array(list(df["description"]))

des_all=" ".join(des)
#splitting the words and counting them (there are punctuations and numbers too)

listed = re.findall(r"[\w']+|[.,!?;]",des_all)

unique,count = np.unique(listed,return_counts=True)

words=dict(zip(unique, count.T))
#sorting everything 

sorted_words=[]

sorted_count=[]

for w in sorted(words, key=words.get, reverse=True):

    sorted_words.append(w)

    sorted_count.append(words[w])
start=12

stop=150

plt.figure(figsize=(10,25))

plt.barh(np.arange(0,len(sorted_count[start:stop])),sorted_count[start:stop],height=0.8,

         tick_label=sorted_words[start:stop],align='center')
feat=np.concatenate(df["features"],axis=0)

feature,count=np.unique(feat,return_counts=True)

feat=dict(zip(unique, count.T))

sorted_feat=[]

sorted_count=[]

for w in sorted(feat, key=feat.get, reverse=True):

    sorted_feat.append(w)

    sorted_count.append(feat[w])
start=0

stop=50

plt.figure(figsize=(10,25))

plt.barh(np.arange(0,len(sorted_count[start:stop])),sorted_count[start:stop],height=0.8,

         tick_label=sorted_feat[start:stop],align='center')
df.corr(method='spearman')
# Adding the columns as we did some feature enginnering in training dataset



df_test["num_photos"]=df_test["photos"].apply(np.size,axis=0)

df_test["num_features"]=df_test["features"].apply(np.size,axis=0)