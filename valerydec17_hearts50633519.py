%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os



print(os.listdir("../input"))
df = pd.read_csv('../input/heart.csv')

df.head(5) # Looking at the data

# Now we know column names
#Checking if any value is NaN

df.isnull().values.any()
def split_train_test(data, valid_ratio, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    valid_set_size = int(len(data) * valid_ratio)

    

    test_indices = shuffled_indices[:test_set_size]

    valid_indices = shuffled_indices[test_set_size:test_set_size+valid_set_size]

    train_indices = shuffled_indices[test_set_size+valid_set_size:]

    return data.iloc[train_indices], data.iloc[valid_indices], data.iloc[test_indices]



train_set, valid_set, test_set = split_train_test(df, 0.2, 0.2)

print(len(train_set), "train +", len(valid_set), "valid +", len(test_set), "test")# importing necessary libraries

from sklearn import datasets

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

 

# loading the iris dataset

iris = datasets.load_iris()





# X -> features, y -> label

X = iris.data

y = iris.target



print (len(X))



# dividing X, y into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

 

# training a Naive Bayes classifier

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X_train, y_train)

gnb_predictions = gnb.predict(X_test)

 

# accuracy on X_test

accuracy = gnb.score(X_test, y_test)

print (accuracy)

 

# creating a confusion matrix

cm = confusion_matrix(y_test, gnb_predictions)
# LIsting all columns

df.dtypes
df.ndim
# Select randomly 5 rows

df.sample(n=5)
df['target'].value_counts() 

# Two categories in the column
data = df

g = sns.PairGrid(data,diag_sharey = False,)





g.map_lower(sns.kdeplot,cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot,lw =3)



g = g.map(plt.scatter)
data = df[['age', 'sex', 'cp', 'trestbps']]

g = sns.PairGrid(data,diag_sharey = False,)





g.map_lower(sns.kdeplot,cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot,lw =3)



g = g.map(plt.scatter)