import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
dataset = pd.read_csv('../input/train.csv')



dataset_test = pd.read_csv('../input/test.csv')



ID=dataset['id']



dataset_test.drop('id',axis=1,inplace=True)



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
print(dataset.head())
dataset = dataset.iloc[:,1:]
print(dataset.describe())
print(dataset.skew())
split = 116

size = 15

data = dataset.iloc[:,split:]

cols = data.columns

n_cols = 2

n_rows = 7



for i in range(n_rows):

    fg,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (12,8))

    for j in range(n_cols):

        sns.violinplot(y=cols[i*2+j],data=dataset,ax=ax[j])
dataset['loss']=np.log1p(dataset['loss'])

sns.violinplot(data=dataset,y=dataset['loss'])

plt.show()
data_corr =data.corr()
data_corr
corr_list = []

for i in range(15):

    for j in range(i+1,15):

        if (data_corr.iloc[i,j] <1 and data_corr.iloc[i,j] >= 0.5) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -0.5):

            corr_list.append([data_corr.iloc[i,j],i,j])
s_corr_list = sorted(corr_list,key = lambda x: abs(x[0]),reverse=True)

for v,i,j in s_corr_list:

    print("%s and %s = %.2f" %  (cols[i],cols[j],v))
for v,i,j in s_corr_list:

    sns.pairplot(dataset,size = 6,x_vars = cols[i],y_vars=cols[j])
type(dataset)
cols =dataset.columns

n_cols= 4 

n_rows = 29

for i in range(29):

    fg,ax = plt.subplots(1,4,figsize=(12,8),sharey=True)

    for j in range(4):

        sns.countplot(x=cols[i*4+j], data=dataset, ax=ax[j])

print(cols)
labels = []

for i in range(0,split):

    train = dataset[cols[i]].unique()

    test = dataset_test[cols[i]].unique()

    labels.append(set(train)|set(test))
del dataset_test
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



cats=[]

for i in range(0,split):

    label_encoder =LabelEncoder()

    label_encoder.fit(labels[i])

    feature = label_encoder.transform(dataset.iloc[:,i])

    feature = feature.reshape(dataset.shape[0], 1)

    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))

    feature = onehot_encoder.fit_transform(feature)

    cats.append(feature)



# Make a 2D array from a list of 1D arrays

encoded_cats = numpy.column_stack(cats)



# Print the shape of the encoded data

print(encoded_cats.shape)



#Concatenate encoded attributes with continuous attributes

dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)

del cats

del feature

del dataset

del encoded_cats

print(dataset_encoded.shape)