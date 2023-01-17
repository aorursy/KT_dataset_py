import pandas as pd

import numpy as np

### Read Training Data and Testing Data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
### Drop id and Seperate loss

train_id = train['id']

test_id =  test['id']

train_label = train['loss']

train.drop('id',axis=1,inplace=True)

train.drop('loss',axis=1,inplace=True)

test.drop('id',axis=1,inplace=True)

print (train.shape, test.shape)
### Column 0-115(index) are categorical, 116 - 129 are numeric

### Create Dummy Variables 

dummies_train = pd.get_dummies(train[train.columns[0:116]])

dummies_test = pd.get_dummies(test[test.columns[0:116]])

print("The shape of train_new and test_new are: " + "%s and %s" % (dummies_train.shape, dummies_test.shape))
### Note we have different size for dummies_train and dummies_test!

### get the columns in train that are not in test

col_to_add = np.setdiff1d(dummies_train.columns, dummies_test.columns)



### add these columns to test, setting them equal to zero

for c in col_to_add:

    dummies_test[c] = 0

### select and reorder the test columns using the train columns

dummies_test = dummies_test[dummies_train.columns]

print("The shape of train_new and test_new are: " + "%s and %s" % (dummies_train.shape, dummies_test.shape))
### Chcek if they have the same columns

mismatch = 0

for i in range(len(dummies_train.columns)):

    if dummies_train.columns[i] != dummies_test.columns[i]:

        mismatch += 1

print("We have %d mismatch." % (mismatch))
### Normalize numeric variables using Min-Max-Scaler

from sklearn import preprocessing

numeric_train = train[train.columns[116:129]]

numeric_test = test[test.columns[116:129]]

min_max_scaler = preprocessing.MinMaxScaler().fit(numeric_train)



### Apply Min-Max-Scalar

train_norm = min_max_scaler.transform(numeric_train)

test_norm = min_max_scaler.transform(numeric_test)



### Convert numeric to dataframe 

train_norm = pd.DataFrame(train_norm,columns=list(numeric_train.columns))

test_norm = pd.DataFrame(test_norm,columns=list(numeric_test.columns))



print("The shape of train_norm and test_norm are: " + "%s and %s" % (train_norm.shape, test_norm.shape))
### Chcek correlation on numeric variables

print(numeric_train.corr())
### Run PCA

from sklearn import decomposition

pca = decomposition.PCA(n_components=6)

pca.fit(train_norm)

train_norm_pca = pca.transform(train_norm)

print(pca.explained_variance_ratio_) 

test_norm_pca = pca.transform(test_norm)

print(pca.explained_variance_ratio_) 

### Convert numeric to dataframe 

train_norm_pca = pd.DataFrame(train_norm_pca,columns=["PCA1","PCA2","PCA3","PCA4","PCA5","PCA6"])

test_norm_pca = pd.DataFrame(test_norm_pca,columns=["PCA1","PCA2","PCA3","PCA4","PCA5","PCA6"])
### Put them back together

train_new = pd.concat((dummies_train,train_norm_pca ),axis=1)

test_new = pd.concat((dummies_test, test_norm_pca),axis=1)

print("The shape of train_new and test_new are: " + "%s and %s" % (train_new.shape, test_new.shape))
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline  

sns.distplot(train_label, color = 'b', hist_kws={'alpha': 0.9}, kde = False)
### Try Log("loss") - way better!

train_label_log = np.log1p(train_label)

sns.distplot(train_label_log, color = 'r', hist_kws={'alpha': 0.9}, kde = False)
### Delete redundant variables to free memory

del dummies_test, dummies_train, numeric_train, numeric_test, train_norm, test_norm
### 80% - 20% training and testing split, random_state=50

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_new, train_label_log, test_size=0.2, random_state=50)