import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
X_train = pd.read_csv('/kaggle/input/titanic/train.csv')

from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer()

X_train_name = X_train["Name"]

X_train_name_dtm = vect.fit_transform(X_train_name)  #This return a sparse matrix.



print(X_train_name_dtm)
# Convert the sparse matrix into an array, followed by converting into a pandas data frame



X_vect = pd.DataFrame(X_train_name_dtm.toarray(), columns=vect.get_feature_names())
X_vect.head()
X_vect.shape
# Create a dataframe showing the count for each name



name_count = pd.DataFrame(X_vect.sum(axis=0).sort_values(ascending=False))

name_count.rename(columns={0 : "Count"})
#To combine X_vect with the original X_train dataframe, you can use the concat method



X_train_combined = pd.concat([X_train, X_vect], axis=1)
X_train_combined.head()