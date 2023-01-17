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
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
contFeatureslist = []

for colName, x in train_data.iloc[1,:].iteritems():

    if(not str(x).isalpha()):

        contFeatureslist.append(colName)

contFeatureslist.remove("id")

contFeatureslist.remove("loss")

catFeatureslist = []

for colName, x in train_data.iloc[1,:].iteritems():

    if(str(x).isalpha()):

        catFeatureslist.append(colName)

        
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.cross_validation import StratifiedKFold
y = train_data['loss']

X = train_data[contFeatureslist+catFeatureslist]
from sklearn.cross_validation import StratifiedKFold

eval_size = 0.10

kf = StratifiedKFold(y, round(1./eval_size))

train_indices, valid_indices = next(iter(kf))
X_train, y_train = X.iloc[valid_indices], y.iloc[valid_indices]

X_valid, y_valid = X.iloc[train_indices], y.iloc[train_indices]
train_categorical_values = np.array(X_train[catFeatureslist])



enc_label = LabelEncoder()

train_data = enc_label.fit_transform(train_categorical_values[:,0])



for i in range(1, train_categorical_values.shape[1]):

    enc_label = LabelEncoder()

    train_data = np.column_stack((train_data, enc_label.fit_transform(train_categorical_values[:,i])))



train_categorical_values = train_data.astype(float)



enc_onehot = OneHotEncoder()

train_cat_data = enc_onehot.fit_transform(train_categorical_values)





cols = [catFeatureslist[i] + '_' + str(j) for i in range(0,len(catFeatureslist)) for j in range(0,enc_onehot.n_values_[i]) ]

train_cat_data_df = pd.DataFrame(train_cat_data.toarray(),columns=cols)



X_train[cols] = train_cat_data_df[cols]
X_train[contFeatureslist+cols].isnull().sum().sum()
from sklearn.feature_selection import chi2, SelectKBest



skb = SelectKBest(chi2, k=100)

skb.fit_transform(X_train[contFeatureslist+cols].fillna(0), np.log1p(y))
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor()

reg.fit(X_train[contFeatureslist+cols], np.log1p(y))