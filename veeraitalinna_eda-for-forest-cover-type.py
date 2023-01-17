import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

%matplotlib inline
print(os.listdir("../input"))
# Load data
train = pd.read_csv('../input/train.csv')
train_X = train[train.columns[1:-1]]
train_Y = train['Cover_Type']
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.head()
train_X.describe()
# Distribution of categories
_ = sns.distplot(train_Y)
fig, axes = plt.subplots(5, 2, figsize=(25, 30))
axes = axes.ravel()
for i, var in enumerate(list(train_X[train_X.columns[:10]])):
    axes[i].hist(train_X[var], bins=30)
    axes[i].set_title(var)
# LDA of the whole dataset
lda = LinearDiscriminantAnalysis()
lda_results = lda.fit_transform(train_X, train_Y)

fig, ax = plt.subplots(figsize=(20,15))
temp = ax.scatter(lda_results[:,0], lda_results[:,1], c=train.Cover_Type)
fig.colorbar(temp)
