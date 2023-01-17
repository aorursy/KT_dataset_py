
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

from sklearn.neural_network import MLPClassifier

#from sklearn.svm import SVC

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns
df = pd.read_json(open("../input/test.json", "r"))

""""""

#df['response'] = 0.

#df.loc[df.interest_level=='medium', 'response'] = 0.5

#df.loc[df.interest_level=='high', 'response'] = 1

#df['mm']=df['response']

""""""
print(df.shape)
#df.head()
df["num_photos"] = df["photos"].apply(len)

df["num_features"] = df["features"].apply(len)

df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

df["created"] = pd.to_datetime(df["created"])

df["created_year"] = df["created"].dt.year

df["created_month"] = df["created"].dt.month

df["created_day"] = df["created"].dt.day
# cheng, if you want to change feature just modify following staff

#num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",

#             "num_photos", "num_features", "num_description_words",

#             "created_year", "created_month", "created_day"]

num_feats = ["bathrooms", "bedrooms", "num_photos","price",'mm']

X = df[num_feats]

y = df["interest_level"]

X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
#clf=SVC(probability=True)

clf = RandomForestClassifier(n_estimators=1000)

#clf=MLPClassifier(hidden_layer_sizes=(10, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

clf.fit(X_train, y_train)

y_val_pred = clf.predict_proba(X_val)

log_loss(y_val, y_val_pred)

