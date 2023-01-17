# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train_x = pd.read_csv("../input/X_train.csv")

df_test_x = pd.read_csv("../input/X_test.csv")

df_train_y = pd.read_csv("../input/y_train.csv")
df_train_x.info()
df_train_y.info()
df_train_x.sample()
df_test_x.series_id.value_counts()
df_train_y.sample(5)
df_train_x.sample(5)
df_train_y[df_train_y.series_id == 1690]
df_train_x[df_train_x.series_id == 1690].shape
"""series_id_y = df_train_y.series_id.tolist()

group_id_y = df_train_y.group_id.tolist()

surface = df_train_y.surface.tolist()

series_id_x = df_train_x.series_id.tolist()

measurment_id = df_train_x.measurement_number.tolist()

just_check_y = list(zip(series_id_y,group_id_y))

just_check_x = list(zip(series_id_x,measurment_id))

store = [-1]*len(just_check_x)

for i in just_check_y:

    if i in just_check_x:

        store[just_check_x.index(i)] = surface[just_check_y.index(i)]

df_train_x["target"] = store"""
df = pd.merge(df_train_x,df_train_y,how='left',on='series_id')
df.sample()
df.surface.value_counts()
df.sample(3)
df.info()
df.sample()
df.drop(columns=["row_id","measurement_number","group_id"], inplace=True)

df.sample(2)
just_check =  df.groupby("series_id").mean().reset_index()

df = pd.merge(just_check,df_train_y,how='left',on='series_id')

df.sample(3)
df.drop(columns=["series_id","group_id"],inplace=True)
df.boxplot()

plt.xticks(rotation = 90)
df[df.columns[-4:]].boxplot()
df.corr()
df.sample(3)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train = scaler.fit_transform(df[df.columns[:-1]])

train_x, test_x, train_y, test_y = train_test_split(train,df[df.columns[-1]],test_size = 0.1)
train_x.shape
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(train_x, train_y)

target = model.predict(test_x)

mat = confusion_matrix(test_y, target)

print("******confusion******\n",mat)
"""from sklearn.metrics import confusion_matrix,accuracy_score

from xgboost import XGBClassifier

for i in range(4,10):

    model = XGBClassifier(model_depth = i)

    model.fit(train_x, train_y)

    target = model.predict(test_x)

    print("accuracy : ", accuracy_score(target, test_y))

mat = confusion_matrix(test_y, target)

print("******confusion******\n",mat)"""
df_test_x.drop(columns=["row_id","measurement_number"],inplace=True)

testing = df_test_x.groupby("series_id").mean().reset_index()

testing.sample(3)

testing.shape
testing.shape
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

fin_train = scale.fit_transform(df[df.columns[:-1]])

test = scale.transform(testing[testing.columns[1:]])
from xgboost import XGBClassifier



model = XGBClassifier()

model.fit(fin_train,df[df.columns[-1]])

target = model.predict(test)
a = testing.series_id.tolist()

b = target

submission = pd.DataFrame({"series_id":a,"surface":b})
submission.surface.value_counts()
#encoding = submission.surface.map({"concrete":1,"soft_pvc":2,"wood":3,"tiled":4,"fine_concrete":5,"soft_tiles":6,"hard_tiles_large_space":7,"carpet":8,"hard_tiles":9})
#submission["encoding"] = encoding
submission.to_csv("submission.csv",index=False)