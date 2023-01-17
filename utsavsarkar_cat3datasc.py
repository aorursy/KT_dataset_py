# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the dataset

dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')

dataset.head()
#describing the data

dataset.describe()
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
print(np.mean(X[0]))

print(np.mean(X[2]))

print(np.mean(X[3]))
print(np.median(X[0]))

print(np.median(X[2]))

print(np.median(X[3]))
from scipy import stats

print(stats.mode(X[2]))
#checking the graph is normal or not

plt.style.use('ggplot')

fig, axis  = plt.subplots(nrows = 2, ncols = 2, figsize = (15,9))

ax0, ax1, ax2, ax3 = axis.flatten()

ax0.hist(dataset['age'])

ax0.set_xlabel('Age')

ax1.hist(dataset['trestbps'])

ax1.set_xlabel('BPS')

ax2.hist(dataset['chol'])

ax2.set_xlabel('Chol')

ax3.hist(dataset['thalach'])

ax3.set_xlabel('thalach')

plt.tight_layout()
#sex vs death

dataset[['sex', 'target']].groupby(['sex'], as_index = False).mean()
#slope vs death

dataset[['slope', 'target']].groupby(['slope'], as_index = False).mean()
#cp vs death

dataset[['cp', 'target']].groupby(['cp'], as_index = False).mean()
#ca vs death

dataset[['ca', 'target']].groupby(['ca'], as_index = False).mean()
#restecg vs death

dataset[['restecg', 'target']].groupby(['restecg'], as_index = False).mean()
#different graph

import seaborn as sns 

g = sns.FacetGrid(dataset, col = 'target')

g.map(plt.hist, 'age')
g = sns.FacetGrid(dataset, col = 'target')

g.map(plt.hist, 'trestbps')
g = sns.FacetGrid(dataset, col = 'target')

g.map(plt.hist, 'chol')
g = sns.FacetGrid(dataset, col = 'target')

g.map(plt.hist, 'thalach')
#checking for correlation

dataset.corr(method='pearson')
#compairing columns 

comparison_column = np.where(dataset["slope"] == dataset["ca"], True, False)

print(comparison_column)
comparison_column = np.where(dataset["restecg"] == dataset["ca"], True, False)

print(comparison_column)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)