import pandas as pd

import numpy as np
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_test = pd.read_csv('../input/test.csv')
df_test.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
num_train = df_train.shape[0]
df_test['loss'] = 0
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
cols = df_all.columns
cols_categorical = cols[1:117].values
df_all.head()
binary =[]

non_binary = []

for i in cols_categorical :

    if  len( np.unique(df_train[i])) ==2 :

        binary.append(i)

    else :

        non_binary.append(i)

    
for i in binary :

    df_all[i] = df_all[i].apply(lambda x : 0 if x=='A' else 1)
df_all.head()
from scipy.sparse import csr_matrix, hstack
for j, i in enumerate(non_binary) :

    if j == 0 :

        feature_non_binary_categorical= csr_matrix( pd.get_dummies(df_all[i], prefix=i,sparse=True ) )

    else :

        feature_non_binary_categorical = hstack((feature_non_binary_categorical, csr_matrix( pd.get_dummies(df_all[i], prefix=i,sparse=True ) )  ))

    

    
feature_binary_categorical = csr_matrix(  df_all[binary].values )
feature_categorical =csr_matrix(  hstack((feature_binary_categorical, feature_non_binary_categorical)) )
feature_categorical_train = feature_categorical[:num_train]
prediction = df_all.loss.values[:num_train]
from sklearn.decomposition import TruncatedSVD
trans = TruncatedSVD(n_components = 2)

feature_categorical_reduced= trans.fit_transform(feature_categorical_train)
plt.scatter(feature_categorical_reduced[:,0],feature_categorical_reduced[:,1], c=prediction, alpha=0.5)
